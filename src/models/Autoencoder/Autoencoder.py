import numpy as np
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Autoencoder(nn.Module):
    def __init__(self, learningRate, weightDecay):
        super().__init__()

        # layer creation
        N_FEATURES = 28 * 28
        N_ENCODED = 28
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, N_ENCODED),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(N_ENCODED, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, N_FEATURES),
            torch.nn.Sigmoid(),
        )
        # defining loss and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate, weight_decay=weightDecay)

    # function that defines feedforward operations
    def forward(self, x):
        x = x.to(device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encodeDecodeImage(self, img):
        img = img.to(device)
        flatImg = img.flatten(start_dim=1)
        with torch.inference_mode():
            modelOutput = self(flatImg)
        modelOutput = modelOutput.reshape(28, 28)
        return modelOutput

    def encodeImage(self, img):
        img = img.to(device)
        flatImg = img.flatten(start_dim=1)
        with torch.inference_mode():
            modelOutput = self.encoder(flatImg)
        return modelOutput

    def trainModel(self, trainLoader, epochs):
        self.train()
        # defining loss and optimizer

        # training loop
        for epoch in range(0, epochs):
            print(f"Epoch {epoch} beginning \n" + "-" * 20)
            # array to store losses so we can calculate the average loss per epoch
            lossArray = np.array([])
            # loop through batches provided by training loader
            for i, data in enumerate(trainLoader):
                features, _ = data
                # since it's a image, it must be flattened
                features = features.to(device)
                features = features.reshape(-1, 28 * 28)

                self.optimizer.zero_grad()

                logits = self(features)
                # We are reconstructing the input
                loss = self.loss_fn(logits, features)
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
