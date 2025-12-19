import numpy as np
import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_classifier(X_train, y_train, input_dim, num_epochs=50, lr=0.01, verbose=False):
    """Train a simple linear classifier."""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    model = SimpleClassifier(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    return model


def extract_model_weights(model):
    """Extract weights and bias from trained model."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().numpy().squeeze()
            W0 = np.array([layer.bias.item()])
            return W, W0
    raise ValueError("No linear layer found in model")


def predict_with_model(model, X):
    """Get predictions and probabilities from model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        probs = model(X_tensor).squeeze().numpy()
        preds = (probs >= 0.5).astype(int)
    return probs, preds
