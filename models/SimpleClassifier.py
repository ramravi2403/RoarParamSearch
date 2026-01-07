import numpy as np
from torch import nn

from models.BaseClassifier import BaseClassifier


class SimpleClassifier(BaseClassifier):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def extract_weights(self):
        linear_layer = self.network[0]
        W = linear_layer.weight.detach().numpy().squeeze()
        W0 = np.array([linear_layer.bias.item()])
        return W, W0
