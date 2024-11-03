import torch
import torch.nn.functional as F
import torch.nn as nn


def conv1d_leakyrelu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = nn.Sequential(
        nn.Conv1d(
            inch,
            outch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm1d(outch),
        nn.LeakyReLU(),
    )
    return convlayer


def conv2d_leakyrelu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = nn.Sequential(
        nn.Conv2d(
            inch,
            outch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(outch),
        nn.LeakyReLU(),
    )
    return convlayer


def mlp(in_features, out_features):
    mlp_layer = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )
    return mlp_layer


class Conv2DRobustClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):  # Add dropout_rate parameter
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=(7, 1),
            stride=(2, 1),
            padding=(3, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(
            dropout_rate
        )  # Add dropout after batch normalization

        self.block1 = ResidualBlock2D(
            64, 128, stride=2, dropout_rate=dropout_rate
        )
        self.block2 = ResidualBlock2D(
            128, 256, stride=2, dropout_rate=dropout_rate
        )

        self.transition = TransitionLayer2D(
            256, 128, pool=True, dropout_rate=dropout_rate
        )

        self.fc1 = nn.Linear(4736, 512)
        self.dropout2 = nn.Dropout(
            dropout_rate
        )  # Add dropout before the fully connected layer
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(2)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)  # Apply dropout after activation
        out = self.block1(out)
        out = self.block2(out)
        out = self.transition(out)
        out = out.view(out.size(0), -1)
        out = self.dropout2(
            F.relu(self.fc1(out))
        )  # Apply dropout after activation
        out = self.fc2(out)
        return out


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout_rate)  # Add dropout layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after activation
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TransitionLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, dropout_rate=0.3):
        super(TransitionLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 convolution for channel reduction
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)  # Include dropout layer
        self.pool = (
            (nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
            if pool
            else nn.Identity()
        )  # Example: Reduce width only

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        # Apply dropout after activation and before pooling
        out = self.pool(out)
        return out


class GMConv1D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack = torch.nn.Sequential(
            conv1d_leakyrelu(1, 256, 16, stride=1),
            nn.Dropout(0.2),
            conv1d_leakyrelu(256, 128, 8),
            nn.Dropout(0.2),
            conv1d_leakyrelu(128, 32, 8),
            nn.Dropout(0.2),
            nn.AvgPool1d(2),
        )

        self.mlp_stack = torch.nn.Sequential(
            mlp(4416, 256),
            mlp(256, 64),
            nn.Linear(in_features=64, out_features=3),
        )

    def forward(self, x):
        batch_size, _, pdf_dim = x.shape
        out_1 = self.conv_stack(x)
        reshaped = out_1.view(batch_size, -1)
        predict_out = self.mlp_stack(reshaped)
        return predict_out


class GMConv1DOptimized(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack = torch.nn.Sequential(
            conv1d_leakyrelu(1, 256, 16, stride=1),
            nn.Dropout(0.3),
            conv1d_leakyrelu(256, 256, 8),
            nn.Dropout(0.3),
            conv1d_leakyrelu(256, 128, 8),
            nn.Dropout(0.3),
            conv1d_leakyrelu(128, 64, 8),
            nn.Dropout(0.3),
            nn.AvgPool1d(2),
        )

        self.mlp_stack = torch.nn.Sequential(
            nn.Linear(8704, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        batch_size, _, _ = x.shape
        out_1 = self.conv_stack(x)
        reshaped = out_1.view(batch_size, -1)
        predict_out = self.mlp_stack(reshaped)
        return predict_out
