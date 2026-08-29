from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """
    Use ConvBNAct because it's the equivalent of Convolutional, Batch Normalization, and Activation layers from TensorFlow
    Modified to use Torch's SiLU activation function.
    SiLU solves dying ReLU dying gradient problem and kept loss surfaces smooth.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        act: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual bottleneck block to preserve gradient flow during deep feature extraction.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNAct(channels // 2, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class BaseDetector(nn.Module, ABC):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        Must return a list of prediction head tensors formatted as:
        List[Tensor(B, H_i, W_i, 5 + num_classes)] where channels represent (x, y, w, h, obj, classes...).
        """
        pass

    def _init_weights(self):
        """
        Kaiming normal initialization for conv layers + focal prior initialization for detection heads.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_detection_head_prior(self, head_conv: nn.Conv2d, prior_prob: float = 0.01):
        """
        Initializes objectness bias to prevent early training instability caused by class imbalance.
        """
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        # Assuming channel layout: [x, y, w, h, obj_logit, cls_logits...]
        with torch.no_grad():
            head_conv.bias.data.fill_(0.0)
            head_conv.bias.data[4:] = bias_value