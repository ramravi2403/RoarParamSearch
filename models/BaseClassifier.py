import torch
from torch import nn


class BaseClassifier(nn.Module):

    def forward(self, x):
        return self.network(x)

    def train_model(self, X_train, y_train, num_epochs=50, lr=0.01, verbose=False):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

        print(f"Training {self.__class__.__name__}.....")

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            preds = self(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = self(X_tensor).squeeze().numpy()
            preds = (probs >= 0.5).astype(int)
        return probs, preds