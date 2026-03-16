import torch.nn as nn
import torch


class SimpleMLPUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class MLPMultipleInputsModel(nn.Module):
    def __init__(self, n_inputs=2000+300, n_neurons=[1024, 1024, 512]):
        super().__init__()
        layers = [
            SimpleMLPUnit(n_in, n_out)
            for n_in, n_out in zip([n_inputs] + n_neurons, n_neurons)
        ] + [nn.Linear(n_neurons[-1], 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, X):
        frags = X.sum(dim=3)
        relative_midpoints = X.sum(dim=2)
        X = torch.concat([frags, relative_midpoints], dim=2)
        X = X.squeeze(1)
        return self.mlp(X).squeeze(1) 
