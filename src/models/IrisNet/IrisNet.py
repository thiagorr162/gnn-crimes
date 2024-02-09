import torch
from torch import nn


# receives 2 tensors
def accuracy(preds, labels):
    nTrue = preds.eq(labels).sum().item()
    return nTrue / preds.size(dim=0)


N_FEATURES = 4
N_OUTPUT = 3
N_FIRSTL = 4


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        # layer creation
        self.layers = nn.Sequential(
            nn.Linear(N_FEATURES, N_OUTPUT),
            # nn.ReLU(),
            # nn.Linear(N_FIRSTL, N_OUTPUT),
        )

    # function that defines feedforward operations
    def forward(self, x):
        return self.layers(x)

    def test(self, X_test, Y_test):
        # there is no need to keep track of the computational graph during test
        with torch.inference_mode():
            loss_fn = nn.CrossEntropyLoss()
            pred = self(X_test)
            loss = loss_fn(pred, Y_test)

        label = torch.argmax(pred, dim=1)
        acc = accuracy(label, Y_test)
        print(f"Accuracy of {acc:.2f} and loss of {loss:.4f}")

    def trainModel(self, X_train, Y_train, epochs, learningRate=0.01, X_test=None, Y_test=None):
        # defining loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learningRate)
        # training loop
        for epoch in range(0, epochs):
            self.train()
            optimizer.zero_grad()
            logits = self(X_train)
            # CrossEntropyLoss already has a softmax layer, that's why we don't pass pred as argument
            loss = loss_fn(logits, Y_train)
            # backwards propagation and optimization
            loss.backward()
            optimizer.step()
            # allows to see how training is affecting model in a ten by ten step
            if X_test is not None and Y_test is not None and epoch % 10 == 0:
                print(f"Epoch:{epoch}")
                self.test(X_test, Y_test)
        # change back to eval mode
        self.eval()
