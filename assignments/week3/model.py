import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    A class to represent a MLP.

    Attributes
    ----------
    input_size : int
        Input dimension
    hidden_size : int
        Size of hidden layer
    num_classes : int
        Number of classes in the data
    hidden_count : int
        Number of hidden layers
    activation : Callable
        Activation function choice
    initializer : Callable
        Initializer choice

    Functions
    -----------
    forward(x):
        Forward pass of the network
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        activation_function = activation()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1568),
            activation_function,
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(1568),
            # torch.nn.Linear(1568, 1568),
            # activation_function,
            # torch.nn.Dropout(p=0.25),
            # torch.nn.BatchNorm1d(1568),
            # torch.nn.Linear(784, 784),
            # activation_function,
            # torch.nn.Dropout(p=0.5),
            # torch.nn.BatchNorm1d(784),
            torch.nn.Linear(1568, 392),
            activation_function,
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(392),
            torch.nn.Linear(392, 98),
            activation_function,
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(98),
            torch.nn.Linear(98, num_classes),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        return self.net(x)
