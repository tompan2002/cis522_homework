import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels, out_channels=9, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(in_channels=9, out_channels=18, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 18, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
