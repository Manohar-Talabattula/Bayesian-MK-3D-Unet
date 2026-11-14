import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout3D(nn.Dropout3d):
    """
    Dropout that remains active during MC inference.
    """
    def forward(self, x):
        return F.dropout3d(
            x,
            self.p,
            training=True,
            inplace=False
        )


class MultiKernelBlock3D(nn.Module):
    """
    Multi-kernel 3D convolution block.

    Uses three convolutional branches with different
    receptive fields and combines their features.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        branch_channels = out_channels // 3
        remaining = out_channels - (branch_channels * 2)

        self.branch1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.InstanceNorm3d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                remaining,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.InstanceNorm3d(remaining),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            MCDropout3D(dropout)
        )

        self.residual = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1, b2, b3], dim=1)

        x = self.fusion(x)

        x = x + self.residual(
            x if x.shape[1] == self.residual.in_channels else
            torch.zeros(
                x.shape[0],
                self.residual.in_channels,
                *x.shape[2:],
                device=x.device,
                dtype=x.dtype
            )
        )

        return F.relu(x, inplace=True)


class EncoderBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        self.block = MultiKernelBlock3D(
            in_channels,
            out_channels,
            dropout
        )

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        features = self.block(x)

        pooled = self.pool(features)

        return features, pooled


class DecoderBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        dropout=0.1
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.block = MultiKernelBlock3D(
            out_channels + skip_channels,
            out_channels,
            dropout
        )

    def forward(self, x, skip):

        x = self.up(x)

        # Handle possible odd spatial dimensions.
        if x.shape[2:] != skip.shape[2:]:

            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        return self.block(x)


class BayesianMKUNet3D(nn.Module):
    """
    Bayesian Multi-Kernel 3D U-Net.

    Stage-1:
        Input  : 1 CT channel
        Output : 13 segmentation classes

    Bayesian approximation:
        Monte-Carlo Dropout.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=13,
        base_channels=16,
        dropout=0.1
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.enc1 = EncoderBlock3D(
            in_channels,
            c1,
            dropout
        )

        self.enc2 = EncoderBlock3D(
            c1,
            c2,
            dropout
        )

        self.enc3 = EncoderBlock3D(
            c2,
            c3,
            dropout
        )

        self.enc4 = EncoderBlock3D(
            c3,
            c4,
            dropout
        )

        # Bottleneck
        self.bottleneck = MultiKernelBlock3D(
            c4,
            c5,
            dropout
        )

        # Decoder
        self.dec4 = DecoderBlock3D(
            c5,
            c4,
            c4,
            dropout
        )

        self.dec3 = DecoderBlock3D(
            c4,
            c3,
            c3,
            dropout
        )

        self.dec2 = DecoderBlock3D(
            c3,
            c2,
            c2,
            dropout
        )

        self.dec1 = DecoderBlock3D(
            c2,
            c1,
            c1,
            dropout
        )

        # Main segmentation head
        self.final_head = nn.Conv3d(
            c1,
            num_classes,
            kernel_size=1
        )

        # Deep supervision heads
        self.aux_head_2 = nn.Conv3d(
            c2,
            num_classes,
            kernel_size=1
        )

        self.aux_head_3 = nn.Conv3d(
            c3,
            num_classes,
            kernel_size=1
        )

        self.aux_head_4 = nn.Conv3d(
            c4,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1, p1 = self.enc1(x)
        e2, p2 = self.enc2(p1)
        e3, p3 = self.enc3(p2)
        e4, p4 = self.enc4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Main output
        main = self.final_head(d1)

        # Auxiliary outputs
        aux2 = self.aux_head_2(d2)
        aux3 = self.aux_head_3(d3)
        aux4 = self.aux_head_4(d4)

        return {
            "main": main,
            "aux2": aux2,
            "aux3": aux3,
            "aux4": aux4
        }

    def enable_mc_dropout(self):
        """
        Enable dropout layers for stochastic inference.
        """

        for module in self.modules():

            if isinstance(module, MCDropout3D):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x,
        n_passes=20
    ):
        """
        Perform multiple stochastic forward passes.

        Returns:
            mean_prediction
            uncertainty
        """

        self.eval()

        self.enable_mc_dropout()

        predictions = []

        for _ in range(n_passes):

            output = self.forward(x)["main"]

            probability = torch.softmax(
                output,
                dim=1
            )

            predictions.append(probability)

        predictions = torch.stack(
            predictions,
            dim=0
        )

        mean_prediction = predictions.mean(
            dim=0
        )

        uncertainty = predictions.var(
            dim=0
        )

        return mean_prediction, uncertainty