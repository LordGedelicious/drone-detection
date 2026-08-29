import torch
import torch.nn as nn
from src.models.base import BaseDetector, ConvBNAct, ResidualBlock


class SingleScaleDetector(BaseDetector):
    """
    Model 1: Baseline Single-Scale Object Detector.
    
    A feed-forward convolutional backbone that progressively downsamples the image 
    to a stride of 32 (P5 stage). Predictions are made exclusively from this single 
    deep grid, serving as the empirical baseline to demonstrate the mathematical 
    limitations of deep spatial downsampling on sub-20-pixel targets.
    """
    def __init__(self, num_classes: int = 1, base_channels: int = 32):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        c = base_channels

        # --- Backbone Stages (Total downsampling: 2^5 = stride 32) ---
        # Input: (B, 3, 640, 640)
        self.stem = ConvBNAct(3, c, kernel_size=3, stride=2, padding=1)           # Stride 2  -> (B, 32, 320, 320)
        
        self.stage2 = nn.Sequential(
            ConvBNAct(c, c * 2, kernel_size=3, stride=2, padding=1),               # Stride 4  -> (B, 64, 160, 160)
            ResidualBlock(c * 2)
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(c * 2, c * 4, kernel_size=3, stride=2, padding=1),           # Stride 8  -> (B, 128, 80, 80)
            ResidualBlock(c * 4),
            ResidualBlock(c * 4)
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(c * 4, c * 8, kernel_size=3, stride=2, padding=1),           # Stride 16 -> (B, 256, 40, 40)
            ResidualBlock(c * 8),
            ResidualBlock(c * 8)
        )
        self.stage5 = nn.Sequential(
            ConvBNAct(c * 8, c * 16, kernel_size=3, stride=2, padding=1),          # Stride 32 -> (B, 512, 20, 20)
            ResidualBlock(c * 16)
        )

        # --- Single-Scale Head ---
        # Channels output: 4 (x, y, w, h) + 1 (objectness) + num_classes
        self.head_channels = 5 + num_classes
        self.head_conv = nn.Sequential(
            ConvBNAct(c * 16, c * 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c * 8, self.head_channels, kernel_size=1, stride=1, padding=0)
        )

        # Initialize weights and focal loss bias prior
        self._init_weights()
        self._init_detection_head_prior(self.head_conv[-1])

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass.
        Returns:
            List containing one tensor of shape (B, H_p5, W_p5, 5 + num_classes)
            ready for DetectionLoss and MetricEvaluator consumption.
        """
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        p5 = self.stage5(x)

        # Output raw logits: (B, 5 + num_classes, H/32, W/32)
        out = self.head_conv(p5)

        # Permute to channels-last layout: (B, H/32, W/32, 5 + num_classes)
        out = out.permute(0, 2, 3, 1).contiguous()

        return [out]


# --- Quick Unit Test & Shape Verification ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SingleScaleDetector(num_classes=1).to(device)
    dummy_input = torch.randn(2, 3, 640, 640, device=device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print("=== Single Scale Baseline Verified ===")
    print(f"Number of output scales: {len(outputs)}")
    print(f"Scale 0 (P5 Grid) shape: {outputs[0].shape}  (Expected: [2, 20, 20, 6])")