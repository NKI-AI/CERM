# A basic MLP for the MNIST dataset.


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A basic MLP for the MNIST dataset."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Parameters
        ----------
        input_dim: int
            input dimension
        output_dim: int
            output dimension
        hidden_dim: int
            hidden dimension
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
