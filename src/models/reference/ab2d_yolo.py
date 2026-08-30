import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseDetector, ConvBNAct, ResidualBlock


# ==========================================
# 1. Attention & Multi-Scale Feature Modules
# ==========================================

class AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction.
    Applies 2D Multi-Head Self-Attention to high-level semantic features (P5).
    """
    def __init__(self, c: int, num_heads: int = 8, hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.c = c

        self.q_proj = nn.Linear(c, c, bias=False)
        self.k_proj = nn.Linear(c, c, bias=False)
        self.v_proj = nn.Linear(c, c, bias=False)
        self.out_proj = nn.Linear(c, c, bias=False)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        self.mlp = nn.Sequential(
            nn.Linear(c, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, c),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W)
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, H*W, C)
        flat_x = x.flatten(2).permute(0, 2, 1)

        # Multi-Head Attention with residual connection
        norm_x = self.norm1(flat_x)
        q = self.q_proj(norm_x).view(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, H * W, C)
        flat_x = flat_x + self.out_proj(attn)

        # MLP Feed-Forward
        flat_x = flat_x + self.mlp(self.norm2(flat_x))

        # Reshape back to (B, C, H, W)
        return flat_x.permute(0, 2, 1).view(B, C, H, W).contiguous()


class DWRBlock(nn.Module):
    """
    Dilation-Wise Residual (DWR) module.
    Extracts multi-scale receptive field features via parallel 3x3 dilated depthwise convolutions.
    """
    def __init__(self, channels: int):
        super().__init__()
        mid_c = channels // 2
        self.conv1 = ConvBNAct(channels, mid_c, kernel_size=1, stride=1, padding=0)

        # Three parallel branches with rates d=1, d=3, d=5
        self.d1 = nn.Conv2d(mid_c // 3, mid_c // 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=mid_c // 3, bias=False)
        self.d2 = nn.Conv2d(mid_c // 3, mid_c // 3, kernel_size=3, stride=1, padding=3, dilation=3, groups=mid_c // 3, bias=False)
        self.d3 = nn.Conv2d(mid_c - 2 * (mid_c // 3), mid_c - 2 * (mid_c // 3), kernel_size=3, stride=1, padding=5, dilation=5, groups=mid_c - 2 * (mid_c // 3), bias=False)

        self.bn = nn.BatchNorm2d(mid_c)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = ConvBNAct(mid_c, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = self.conv1(x)
        c_split = feat.shape[1] // 3
        f1, f2, f3 = feat[:, :c_split], feat[:, c_split:2 * c_split], feat[:, 2 * c_split:]
        
        out = torch.cat([self.d1(f1), self.d2(f2), self.d3(f3)], dim=1)
        out = self.conv2(self.act(self.bn(out)))
        return residual + out


class C2fDWR(nn.Module):
    """C2f module integrated with DWR bottleneck blocks."""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        self.c = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBNAct((2 + num_blocks) * self.c, out_channels, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([DWRBlock(self.c) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, 1))


class BiFPNNode(nn.Module):
    """Weighted Fast Normalized Feature Fusion for BiFPN."""
    def __init__(self, channels: int, num_inputs: int = 2, eps: float = 1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps
        self.conv = ConvBNAct(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: list) -> torch.Tensor:
        w = F.relu(self.weights)
        w_sum = torch.sum(w) + self.eps
        fused = sum((w[i] / w_sum) * inputs[i] for i in range(len(inputs)))
        return self.conv(fused)


# ==========================================
# 2. Complete AB2D-YOLO Architecture
# ==========================================

class AB2DYOLO(BaseDetector):
    """
    AB2D-YOLO: Drone Small-Object Detection Architecture.
    
    Backbone:
      - 5-stage downsampling (P1 -> P5)
      - AIFI at P5 stage
    Neck:
      - 4-Scale BiFPN feature fusion (P2, P3, P4, P5)
      - C2f_DWR multi-scale contextual blocks
    Head:
      - Decoupled detection heads outputting at 4 spatial scales (stride 4, 8, 16, 32)
    """
    def __init__(self, num_classes: int = 1, base_channels: int = 32):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        c = base_channels  # Base width: 32

        # --- Backbone ---
        self.stem = ConvBNAct(3, c, kernel_size=3, stride=2, padding=1)             # /2  (B, 32, 320, 320)
        
        self.stage2 = nn.Sequential(                                                # /4  (B, 64, 160, 160) - P2
            ConvBNAct(c, c * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock(c * 2)
        )
        self.stage3 = nn.Sequential(                                                # /8  (B, 128, 80, 80)  - P3
            ConvBNAct(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(c * 4),
            ResidualBlock(c * 4)
        )
        self.stage4 = nn.Sequential(                                                # /16 (B, 256, 40, 40)  - P4
            ConvBNAct(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            ResidualBlock(c * 8),
            ResidualBlock(c * 8)
        )
        self.stage5 = nn.Sequential(                                                # /32 (B, 512, 20, 20)  - P5
            ConvBNAct(c * 8, c * 16, kernel_size=3, stride=2, padding=1),
            ResidualBlock(c * 16)
        )
        # Replace SPPF with Attention-based Intra-scale Feature Interaction (AIFI)
        self.aifi = AIFI(c * 16, num_heads=8, hidden_dim=c * 16)

        # --- Channel Reducers for BiFPN Neck (Unified channels: c * 4 = 128) ---
        neck_c = c * 4
        self.lat_p5 = ConvBNAct(c * 16, neck_c, kernel_size=1, stride=1, padding=0)
        self.lat_p4 = ConvBNAct(c * 8, neck_c, kernel_size=1, stride=1, padding=0)
        self.lat_p3 = ConvBNAct(c * 4, neck_c, kernel_size=1, stride=1, padding=0)
        self.lat_p2 = ConvBNAct(c * 2, neck_c, kernel_size=1, stride=1, padding=0)

        # --- Top-Down Path (BiFPN) ---
        self.td_p4 = BiFPNNode(neck_c, num_inputs=2)
        self.td_p3 = BiFPNNode(neck_c, num_inputs=2)
        self.td_p2 = BiFPNNode(neck_c, num_inputs=2)

        # --- Bottom-Up Path (BiFPN + C2f_DWR) ---
        self.c2f_dwr2 = C2fDWR(neck_c, neck_c, num_blocks=1)
        self.bu_p3 = BiFPNNode(neck_c, num_inputs=3)
        self.c2f_dwr3 = C2fDWR(neck_c, neck_c, num_blocks=1)
        self.bu_p4 = BiFPNNode(neck_c, num_inputs=3)
        self.c2f_dwr4 = C2fDWR(neck_c, neck_c, num_blocks=1)
        self.bu_p5 = BiFPNNode(neck_c, num_inputs=2)
        self.c2f_dwr5 = C2fDWR(neck_c, neck_c, num_blocks=1)

        self.downsample2 = ConvBNAct(neck_c, neck_c, kernel_size=3, stride=2, padding=1)
        self.downsample3 = ConvBNAct(neck_c, neck_c, kernel_size=3, stride=2, padding=1)
        self.downsample4 = ConvBNAct(neck_c, neck_c, kernel_size=3, stride=2, padding=1)

        # --- 4-Scale Detection Heads ---
        # Channels: 4 (x, y, w, h) + 1 (objectness) + num_classes
        self.head_channels = 5 + num_classes
        self.head_p2 = nn.Conv2d(neck_c, self.head_channels, kernel_size=1)
        self.head_p3 = nn.Conv2d(neck_c, self.head_channels, kernel_size=1)
        self.head_p4 = nn.Conv2d(neck_c, self.head_channels, kernel_size=1)
        self.head_p5 = nn.Conv2d(neck_c, self.head_channels, kernel_size=1)

        # Weight initialization and focal loss prior
        self._init_weights()
        for head in [self.head_p2, self.head_p3, self.head_p4, self.head_p5]:
            self._init_detection_head_prior(head)

    def forward(self, x: torch.Tensor) -> list:
        # --- Backbone Forward ---
        x = self.stem(x)
        p2_in = self.stage2(x)                     # (B, 64, 160, 160)
        p3_in = self.stage3(p2_in)                 # (B, 128, 80, 80)
        p4_in = self.stage4(p3_in)                 # (B, 256, 40, 40)
        p5_in = self.aifi(self.stage5(p4_in))      # (B, 512, 20, 20)

        # Align lateral channels
        p2 = self.lat_p2(p2_in)                    # (B, 128, 160, 160)
        p3 = self.lat_p3(p3_in)                    # (B, 128, 80, 80)
        p4 = self.lat_p4(p4_in)                    # (B, 128, 40, 40)
        p5 = self.lat_p5(p5_in)                    # (B, 128, 20, 20)

        # --- Top-Down Pyramid ---
        td4 = self.td_p4([p4, F.interpolate(p5, scale_factor=2, mode="nearest")])
        td3 = self.td_p3([p3, F.interpolate(td4, scale_factor=2, mode="nearest")])
        td2 = self.td_p2([p2, F.interpolate(td3, scale_factor=2, mode="nearest")])

        # --- Bottom-Up Pyramid with C2f_DWR ---
        out_p2 = self.c2f_dwr2(td2)
        
        bu3 = self.bu_p3([p3, td3, self.downsample2(out_p2)])
        out_p3 = self.c2f_dwr3(bu3)

        bu4 = self.bu_p4([p4, td4, self.downsample3(out_p3)])
        out_p4 = self.c2f_dwr4(bu4)

        bu5 = self.bu_p5([p5, self.downsample4(out_p4)])
        out_p5 = self.c2f_dwr5(bu5)

        # --- Prediction Heads (Fine to Coarse: P2, P3, P4, P5) ---
        preds = [
            self.head_p2(out_p2).permute(0, 2, 3, 1).contiguous(),  # (B, 160, 160, 5 + C)
            self.head_p3(out_p3).permute(0, 2, 3, 1).contiguous(),  # (B, 80, 80, 5 + C)
            self.head_p4(out_p4).permute(0, 2, 3, 1).contiguous(),  # (B, 40, 40, 5 + C)
            self.head_p5(out_p5).permute(0, 2, 3, 1).contiguous()   # (B, 20, 20, 5 + C)
        ]

        return preds


# --- Quick Unit Test & Shape Verification ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AB2DYOLO(num_classes=1, base_channels=32).to(device)
    dummy_input = torch.randn(2, 3, 640, 640, device=device)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("=== AB2D-YOLO Output Verification ===")
    print(f"Total scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Scale {i} shape: {out.shape}")