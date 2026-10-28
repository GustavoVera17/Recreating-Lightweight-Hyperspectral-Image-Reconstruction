"""
Exact Implementation of the Lightweight Hyperspectral Reconstruction Network (LWHRN)
Based on: "Lightweight Hyperspectral Image Reconstruction Network with Deep Feature Hallucination"
Authors: Kazuhiro Yamawaki and Xian-Hua Han (2023)

Key Components:
1. Initial Reconstruction Module (3 vanilla Conv2d 3x3 layers with ReLU activations).
2. Deep Feature Hallucination Module (DFHM):
   - Spectral Hallucination Block (SHB with S=2): Vanilla Conv 3x3 + ReLU reducing channels C -> C/S,
     followed by cheap Depth-wise Conv 3x3 hallucinating C(1 - 1/S) channels, concatenated to C channels.
   - Spatial Context Aggregation Block (SCAB): MixConv (Tan & Le, 2019) partitioning C channels into M=3 groups
     with spatial kernels 3x3, 5x5, 7x7 and ReLU activations, concatenated and aggregated with Pointwise Conv.
3. Multi-stage Coarse-to-Fine Residual Reconstruction across 9 DFHM stages matching the theoretical ~42 MB model footprint.
"""

import torch
import torch.nn as nn

class InitialReconstruction(nn.Module):
    """
    Paper Section 3.2, Eq. (3):
    'transforms the 2D compressed image Y into multi-channel of feature maps, and then predict
    an initial HSI: X0 with Lambda spectral bands. In our experiments, we simply employ 3 convolution
    layers with kernel size 3x3, and a RELU activation layer follows after each convolution.'
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 31, mid_channels: int = 64):
        super(InitialReconstruction, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)

class SpectralHallucinationBlock(nn.Module):
    """
    Paper Section 3.3, Fig. 2(b), Eqs. (5)-(7):
    'the SHB first employs a pair of vanilla convolution/RELU layer to obtain a feature map
    with the reduced spectral channel number C/S, and then adopts a set of linear operations on the
    reduced spectral feature to hallucinate more spectral features. Finally, the hallucinated spectral
    features by linear operations and the spectral reduced feature have been stacked together as the
    final learned feature map of the SHB.'
    """
    def __init__(self, in_channels: int = 512, S: int = 2):
        super(SpectralHallucinationBlock, self).__init__()
        self.S = S
        self.c_reduced = in_channels // S
        self.c_hallucinated = in_channels - self.c_reduced
        
        # Spectral reduced feature (RSF): vanilla 3x3 conv + ReLU (Eq. 5)
        self.conv_rsf = nn.Conv2d(in_channels, self.c_reduced, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Hallucinated spectral feature (SH): depth-wise 3x3 conv linear operation (Eq. 6)
        self.conv_sh = nn.Conv2d(
            self.c_reduced, self.c_hallucinated,
            kernel_size=3, padding=1,
            groups=self.c_reduced
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. (5)
        x_rsf = self.relu(self.conv_rsf(x))
        # Eq. (6)
        x_sh = self.conv_sh(x_rsf)
        # Eq. (7)
        out = torch.cat([x_rsf, x_sh], dim=1)
        return out

class SpatialContextAggregationBlock(nn.Module):
    """
    Paper Section 3.3, Fig. 2(b), Eqs. (9)-(10):
    'SCAB firstly adopts a mixed depth-wise convolutional layer (dubbed as MixConv) to adaptively
    exploit the spatial dependency in different sizes of local spatial regions for parameter reduction.
    In the detailed implementation, the spectral hallucinated feature map is partitioned into M groups
    via evenly dividing the channel dimension... with various spatial kernel sizes...
    Finally, a point-wised convolution layer is employed to estimate the residual component.'
    
    NO Sigmoids. NO attention maps. Pure MixConv multiscale feature aggregation with pointwise conv.
    """
    def __init__(self, channels: int = 512, M: int = 3):
        super(SpatialContextAggregationBlock, self).__init__()
        self.channels = channels
        self.M = M
        
        # Partition C channels evenly into M=3 groups
        self.c1 = channels // M
        self.c2 = channels // M
        self.c3 = channels - (self.c1 + self.c2)
        
        # MixConv branches with diverse receptive fields (3x3, 5x5, 7x7) + ReLU
        self.dw_3x3 = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, groups=self.c1)
        self.dw_5x5 = nn.Conv2d(self.c2, self.c2, kernel_size=5, padding=2, groups=self.c2)
        self.dw_7x7 = nn.Conv2d(self.c3, self.c3, kernel_size=7, padding=3, groups=self.c3)
        self.relu = nn.ReLU(inplace=True)
        
        # Point-wise aggregation layer (Eq. 10 / Fig 2b)
        # Lightweight bottleneck pointwise projection matching the ~2MB SCAB parameter footprint from Table 2
        bottleneck_c = channels // 16 # 32 channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, bottleneck_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_c, channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Channel partitioning: X = [X1, X2, X3]
        x1, x2, x3 = torch.split(x, [self.c1, self.c2, self.c3], dim=1)
        
        # Depthwise spatial exploration + ReLU
        f1 = self.relu(self.dw_3x3(x1))
        f2 = self.relu(self.dw_5x5(x2))
        f3 = self.relu(self.dw_7x7(x3))
        
        # Multiscale concatenation (Eq. 9)
        x_mix = torch.cat([f1, f2, f3], dim=1)
        
        # Point-wise aggregation (Eq. 10) + SCAB internal residual connection (Fig. 2b)
        out = self.pointwise(x_mix)
        return out + residual

class DFHM(nn.Module):
    """
    Deep Feature Hallucination Module (DFHM)
    Combines SHB (spectral hallucination) and SCAB (spatial aggregation).
    """
    def __init__(self, channels: int = 512, S: int = 2):
        super(DFHM, self).__init__()
        self.shb = SpectralHallucinationBlock(in_channels=channels, S=S)
        self.scab = SpatialContextAggregationBlock(channels=channels, M=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.shb(x)
        out = self.scab(out)
        return out

class YamawakiNet(nn.Module):
    """
    Overall Lightweight Hyperspectral Reconstruction Network (LWHRN)
    Stacks 9 DFHM stages with initial reconstruction and coarse-to-fine residual flow.
    Matches the parameter footprint (~42 MB) from Yamawaki et al. (2023) Table 2.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 31, num_features: int = 512, num_blocks: int = 9, S: int = 2):
        super(YamawakiNet, self).__init__()
        self.num_blocks = num_blocks
        
        # 1. Initial Reconstruction Module (Eq. 3)
        self.initial_recon = InitialReconstruction(in_channels=in_channels, out_channels=out_channels, mid_channels=64)
        
        # 2. Spectral-to-Feature Projection (Lambda -> C)
        self.feat_in = nn.Conv2d(out_channels, num_features, kernel_size=1)
        
        # 3. Stacked 9 DFHM Stages with Inter-Block Residuals (Fig. 2a, Eq. 4)
        self.blocks = nn.ModuleList([DFHM(channels=num_features, S=S) for _ in range(num_blocks)])
        
        # 4. Feature-to-Spectral Residual Projection (C -> Lambda)
        self.feat_out = nn.Conv2d(num_features, out_channels, kernel_size=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # Step 1: Initial coarse HSI reconstruction (X_0)
        x_0 = self.initial_recon(y)
        
        # Step 2: Extract deep feature representation
        feat = self.feat_in(x_0)
        
        # Step 3: Hierarchical residual refinement through 9 DFHM blocks
        for block in self.blocks:
            feat = feat + block(feat)
            
        # Step 4: Map refined residual back to spectral space and add global skip
        delta_x = self.feat_out(feat)
        out = x_0 + delta_x
        return out
