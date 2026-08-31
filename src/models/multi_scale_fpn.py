import torch
import torch.nn as nn
from src.models.base import BaseDetector, ConvBNAct, ResidualBlock


class MultiScaleFPNDetector(BaseDetector):
    """
    Model 2: Multi-Scale Feature Pyramid Network (FPN) Object Detector.
    
    Fuses deep semantic feature maps with high-resolution shallow feature maps 
    via top-down pathways and lateral 1x1 convolutions. Outputs predictions from 
    three distinct pyramid scales: P3 (stride 8), P4 (stride 16), and P5 (stride 32).
    """
    def __init__(self, num_classes: int = 1, base_channels: int = 32, fpn_out_channels: int = 128):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        c = base_channels
        fpn_c = fpn_out_channels

        # --- Backbone Stages (Identical to Model 1 for fair ablation) ---
        self.stem = ConvBNAct(3, c, kernel_size=3, stride=2, padding=1)           # Stride 2  -> (B, 32, 320, 320)
        
        self.stage2 = nn.Sequential(
            ConvBNAct(c, c * 2, kernel_size=3, stride=2, padding=1),               # Stride 4  -> (B, 64, 160, 160)
            ResidualBlock(c * 2)
        )
        self.stage3 = nn.Sequential(                                              # C3 (Stride 8)
            ConvBNAct(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(c * 4),
            ResidualBlock(c * 4)
        )
        self.stage4 = nn.Sequential(                                              # C4 (Stride 16)
            ConvBNAct(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            ResidualBlock(c * 8),
            ResidualBlock(c * 8)
        )
        self.stage5 = nn.Sequential(                                              # C5 (Stride 32)
            ConvBNAct(c * 8, c * 16, kernel_size=3, stride=2, padding=1),
            ResidualBlock(c * 16)
        )

        # --- FPN Lateral Projections (1x1 Convolutions) ---
        self.lateral_c5 = nn.Conv2d(c * 16, fpn_c, kernel_size=1, stride=1, padding=0)
        self.lateral_c4 = nn.Conv2d(c * 8, fpn_c, kernel_size=1, stride=1, padding=0)
        self.lateral_c3 = nn.Conv2d(c * 4, fpn_c, kernel_size=1, stride=1, padding=0)

        # --- FPN Smooth Convolutions (3x3 Convolutions to reduce aliasing) ---
        self.smooth_p4 = ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1)
        self.smooth_p3 = ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1)

        # Top-down upsampling operator (nearest neighbor)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # --- Multi-Scale Detection Heads ---
        # Channels per cell: 4 (x, y, w, h) + 1 (objectness) + num_classes
        self.head_channels = 5 + num_classes

        self.head_p3 = nn.Sequential(
            ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(fpn_c, self.head_channels, kernel_size=1, stride=1, padding=0)
        )
        self.head_p4 = nn.Sequential(
            ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(fpn_c, self.head_channels, kernel_size=1, stride=1, padding=0)
        )
        self.head_p5 = nn.Sequential(
            ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(fpn_c, self.head_channels, kernel_size=1, stride=1, padding=0)
        )

        # Initialize weights and focal loss bias priors across all three heads
        self._init_weights()
        self._init_detection_head_prior(self.head_p3[-1])
        self._init_detection_head_prior(self.head_p4[-1])
        self._init_detection_head_prior(self.head_p5[-1])

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass.
        Returns:
            List of 3 tensors sorted from finest to coarsest:
            [
                Tensor(B, H/8,  W/8,  5 + num_classes),  # P3 Head (Fine / Small targets)
                Tensor(B, H/16, W/16, 5 + num_classes),  # P4 Head (Mid)
                Tensor(B, H/32, W/32, 5 + num_classes)   # P5 Head (Coarse / Large targets)
            ]
        """
        p3, p4, p5 = self.neck_forward(x)
        out_p3 = self.head_p3(p3).permute(0, 2, 3, 1).contiguous()
        out_p4 = self.head_p4(p4).permute(0, 2, 3, 1).contiguous()
        out_p5 = self.head_p5(p5).permute(0, 2, 3, 1).contiguous()
        return [out_p3, out_p4, out_p5]

    neck_channels = [128, 128, 128]  # fpn_out_channels, x3

    def neck_forward(self, x: torch.Tensor) -> list:
        """Bottom-up backbone + top-down FPN -> [P3, P4, P5] feature maps
        (finest to coarsest) as fed to the three detection heads."""
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p5 = self.lateral_c5(c5)
        p4 = self.smooth_p4(self.lateral_c4(c4) + self.upsample(p5))
        p3 = self.smooth_p3(self.lateral_c3(c3) + self.upsample(p4))
        return [p3, p4, p5]


# --- Quick Unit Test & Shape Verification ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiScaleFPNDetector(num_classes=1).to(device)
    dummy_input = torch.randn(2, 3, 640, 640, device=device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print("=== Multi-Scale FPN Verified ===")
    print(f"Number of output scales: {len(outputs)}")
    print(f"Scale 0 (P3 Grid - Fine)   shape: {outputs[0].shape}  (Expected: [2, 80, 80, 6])")
    print(f"Scale 1 (P4 Grid - Mid)    shape: {outputs[1].shape}  (Expected: [2, 40, 40, 6])")
    print(f"Scale 2 (P5 Grid - Coarse) shape: {outputs[2].shape}  (Expected: [2, 20, 20, 6])")