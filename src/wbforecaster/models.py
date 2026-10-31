"""Convolutional forecast models on the 64x32 lat/lon grid."""
import torch.nn as nn


class PeriodicConv2d(nn.Module):
    """Conv2d with circular padding in longitude (the globe wraps) and replicate padding in latitude."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        # x: (B, C, Lat, Lon)
        x = nn.functional.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        x = nn.functional.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        return self.conv(x)


class LinearRegressionModel(nn.Module):
    """A single 5x5 convolution with no activation: a local linear map from x_t to x_{t+1}."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = PeriodicConv2d(in_channels, out_channels, kernel_size=5)

    def forward(self, x):
        return self.linear(x)


class SimpleWeatherNet(nn.Module):
    """Three-layer residual CNN: predicts the change x_{t+1} - x_t.

    The input is the state followed by any forcing channels, so in_channels >= out_channels.
    The residual adds the change to the state channels only: forcings are inputs, never outputs.
    """

    def __init__(self, in_channels, out_channels, hidden_dim=64, kernel_size=3, dilation=1):
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            PeriodicConv2d(in_channels, hidden_dim, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            PeriodicConv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            PeriodicConv2d(hidden_dim, out_channels, kernel_size=kernel_size, dilation=dilation),
        )

    def forward(self, x):
        return x[:, : self.out_channels] + self.net(x)


class ComplexWeatherNet(nn.Module):
    """ResNet-style CNN with a dilated block for a larger receptive field.

    Like SimpleWeatherNet, the residual adds the change to the state channels only.
    """

    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.out_channels = out_channels
        self.gelu = nn.GELU()
        self.in_layer = PeriodicConv2d(in_channels, hidden_dim)
        self.res1_1 = PeriodicConv2d(hidden_dim, hidden_dim)
        self.res1_2 = PeriodicConv2d(hidden_dim, hidden_dim)
        self.res2_1 = PeriodicConv2d(hidden_dim, hidden_dim, dilation=2)
        self.res2_2 = PeriodicConv2d(hidden_dim, hidden_dim, dilation=2)
        self.out_layer = PeriodicConv2d(hidden_dim, out_channels)

    def forward(self, x):
        feat = self.gelu(self.in_layer(x))
        feat = self.gelu(feat + self.res1_2(self.gelu(self.res1_1(feat))))
        feat = self.gelu(feat + self.res2_2(self.gelu(self.res2_1(feat))))
        return x[:, : self.out_channels] + self.out_layer(feat)


MODELS = {
    "linear": LinearRegressionModel,
    "simple": SimpleWeatherNet,
    "complex": ComplexWeatherNet,
}
