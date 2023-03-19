import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    CNN model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initializes layers of model

        Arguments:
            num_channels (int): Number of channels of input, num_classes (int): Number of possible classes.

        Returns:
            Nothing.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels, out_channels=12, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(in_channels=12, out_channels=21, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(21),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 21, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes inputs through model and returns probabilities of classes

        Arguments:
            x (torch.Tensor): Image to classify

        Returns:
            Tensor of probabilities of each class for the image
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
