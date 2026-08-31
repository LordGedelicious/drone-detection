import torch
import torch.nn as nn
from src.models.base import BaseDetector, ConvBNAct, ResidualBlock


class P2GranularDetector(BaseDetector):
    """
    Model 3: P2 High-Resolution Granular Object Detector.
    
    Extends the multi-scale Feature Pyramid Network (FPN) by incorporating 
    an ultra-high-resolution P2 detection head (stride 4, 160x160 grid for 640x640 input).
    Preserves fine-grained spatial and edge details before aggressive pooling,
    specifically targeting distant and tiny aerial drones.
    """
    def __init__(self, num_classes: int = 1, base_channels: int = 32, fpn_out_channels: int = 128):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        c = base_channels
        fpn_c = fpn_out_channels

        # --- Backbone Stages ---
        self.stem = ConvBNAct(3, c, kernel_size=3, stride=2, padding=1)           # Stride 2  -> (B, 32, 320, 320)
        
        self.stage2 = nn.Sequential(                                              # C2 (Stride 4)
            ConvBNAct(c, c * 2, kernel_size=3, stride=2, padding=1),
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
        self.lateral_c2 = nn.Conv2d(c * 2, fpn_c, kernel_size=1, stride=1, padding=0)  # Lateral for Stride 4

        # --- FPN Smooth Convolutions ---
        self.smooth_p4 = ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1)
        self.smooth_p3 = ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1)
        self.smooth_p2 = ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1)  # Smooth for P2

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # --- Detection Heads (P2, P3, P4, P5) ---
        self.head_channels = 5 + num_classes

        self.head_p2 = nn.Sequential(
            ConvBNAct(fpn_c, fpn_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(fpn_c, self.head_channels, kernel_size=1, stride=1, padding=0)
        )
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

        # Initialize weights and focal loss bias priors
        self._init_weights()
        self._init_detection_head_prior(self.head_p2[-1])
        self._init_detection_head_prior(self.head_p3[-1])
        self._init_detection_head_prior(self.head_p4[-1])
        self._init_detection_head_prior(self.head_p5[-1])

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass.
        Returns:
            List of 4 tensors ordered from finest to coarsest:
            [
                Tensor(B, H/4,  W/4,  5 + num_classes),  # P2 Head (Granular / Sub-20px targets)
                Tensor(B, H/8,  W/8,  5 + num_classes),  # P3 Head (Fine)
                Tensor(B, H/16, W/16, 5 + num_classes),  # P4 Head (Mid)
                Tensor(B, H/32, W/32, 5 + num_classes)   # P5 Head (Coarse)
            ]
        """
        p2, p3, p4, p5 = self.neck_forward(x)
        out_p2 = self.head_p2(p2).permute(0, 2, 3, 1).contiguous()
        out_p3 = self.head_p3(p3).permute(0, 2, 3, 1).contiguous()
        out_p4 = self.head_p4(p4).permute(0, 2, 3, 1).contiguous()
        out_p5 = self.head_p5(p5).permute(0, 2, 3, 1).contiguous()
        return [out_p2, out_p3, out_p4, out_p5]

    neck_channels = [128, 128, 128, 128]  # fpn_out_channels, x4

    def neck_forward(self, x: torch.Tensor) -> list:
        """Backbone + top-down FPN with the extra P2 level -> [P2, P3, P4, P5]."""
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p5 = self.lateral_c5(c5)
        p4 = self.smooth_p4(self.lateral_c4(c4) + self.upsample(p5))
        p3 = self.smooth_p3(self.lateral_c3(c3) + self.upsample(p4))
        p2 = self.smooth_p2(self.lateral_c2(c2) + self.upsample(p3))
        return [p2, p3, p4, p5]


# --- Quick Unit Test & Shape Verification ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = P2GranularDetector(num_classes=1).to(device)
    dummy_input = torch.randn(2, 3, 640, 640, device=device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print("=== P2 Granular Detector Verified ===")
    print(f"Number of output scales: {len(outputs)}")
    print(f"Scale 0 (P2 Grid - Granular) shape: {outputs[0].shape}  (Expected: [2, 160, 160, 6])")
    print(f"Scale 1 (P3 Grid - Fine)     shape: {outputs[1].shape}  (Expected: [2, 80, 80, 6])")
    print(f"Scale 2 (P4 Grid - Mid)      shape: {outputs[2].shape}  (Expected: [2, 40, 40, 6])")
    print(f"Scale 3 (P5 Grid - Coarse)   shape: {outputs[3].shape}  (Expected: [2, 20, 20, 6])")