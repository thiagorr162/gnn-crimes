import numpy as np
import torch
from torch import nn

BATCH_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# receives 2 tensors and returns how many elements are equal
def countEqual(preds, labels):
    preds = preds.to(device)
    labels = labels.to(device)
    nEqual = preds.eq(labels).sum().item()
    return nEqual


class GRUSentimentAnalysis(nn.Module):
    def __init__(self, learningRate, embeddingDim, hiddenDim, vocabSize, bidirectional, nLayers, paddingIdx):
        super().__init__()
        self.embeddingLayer = nn.Embedding(vocabSize, embeddingDim, paddingIdx, device=device)
        self.gru = nn.GRU(
            embeddingDim,
            hiddenDim,
            device=device,
            batch_first=True,
            num_layers=nLayers,
            bidirectional=bidirectional,
            dropout=0.3,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=hiddenDim * 2 if bidirectional else hiddenDim, out_features=1, device=device),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(0.3)
        # defining loss and optimizer
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)

    # function that defines feedforward operations
    def forward(self, x):
        x = x.to(device)
        xEmbed = self.embeddingLayer(x)
        xEmbed = self.dropout(xEmbed)
        output, hn = self.gru(xEmbed)
        # it only matters the last output of the sequence
        output = output[:, -1]
        output = self.dropout(output)
        # input: tensor shape (batch), sequenceL, inputSize
        return self.fc(output)

    def test(self, testLoader):
        # there is no need to keep track of the computational graph during test
        totalSamples = 0
        correctGuesses = 0
        with torch.no_grad():
            for i, data in enumerate(testLoader):
                features = data["ids"].to(device)
                label = data["label"].to(device)
                totalSamples += len(features)
                logits = self(features)
                predictions = torch.round(logits)
                correctGuesses += countEqual(predictions, label)

        acc = correctGuesses / totalSamples
        print(f"Accuracy of {acc:.4f} ")

    def trainModel(self, trainLoader, epochs):
        self.train()
        # training loop
        for epoch in range(0, epochs):
            print(f"Epoch {epoch} beginning \n" + "-" * 20)
            # array to store losses so we can calculate the average loss per epoch
            lossArray = np.array([])
            # loop through batches provided by training loader
            for i, data in enumerate(trainLoader):
                features = data["ids"].to(device)
                label = data["label"].to(device)
                self.optimizer.zero_grad()
                logits = self(features)
                loss = self.loss_fn(logits, label)
                # backwards propagation and optimization
                loss.backward()
                self.optimizer.step()
                # checks loss
                if i % 20 == 0:
                    print(f"Batch:{i}, Epoch{epoch}, Loss:{loss}")
                    # self.test(X_test, Y_test)
                lossArray = np.append(lossArray, loss.item())
            print(f"Epoch {epoch} has finished, avg loss: {np.mean(lossArray)} \n" + "-" * 20)
        # change back to eval mode
        self.eval()
