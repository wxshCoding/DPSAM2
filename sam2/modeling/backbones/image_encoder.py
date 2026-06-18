# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .MFB import MFB_modified
# from MFB import MFB_modified


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        # features, pos = self.neck(self.trunk(sample))
        tr = self.trunk(sample)
        output_features, position_encodings = self.neck(tr)
        if self.scalp > 0:
            # Discard the lowest resolution features
            output_features = output_features[: -self.scalp]
            position_encodings = position_encodings[: -self.scalp]

        src = output_features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": position_encodings,
            "backbone_fpn": output_features,
            "mfb_output_features": list(output_features),
            "mfb_position_encodings": list(position_encodings),
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos

class MFBFpnNeck(nn.Module):
    """
    A modified FPN neck using RFB modules for feature refinement
    """
    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.position_encoding = position_encoding
        self.mrb_convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        
        # Replace simple convs with RFB modules
        for dim in backbone_channel_list:
            self.mrb_convs.append(MFB_modified(dim, d_model))
            
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # Set up top-down pathway levels
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.mrb_convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.mrb_convs)
        pos = [None] * len(self.mrb_convs)
        assert len(xs) == len(self.mrb_convs)

        # FPN forward pass with RFB modules
        prev_features = None
        n = len(self.mrb_convs) - 1
        
        # Top-down pathway
        for i in range(n, -1, -1):
            x = xs[i]
            # Replace conv with RFB module
            lateral_features = self.mrb_convs[n - i](x)
            
            # Combine features if in top-down levels
            if i in self.fpn_top_down_levels and prev_features is not None:
                # Upsample previous features
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                # Combine lateral and top-down features
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
                
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
    


def test_neck():
    import torch
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    # 1. Create sample input features
    batch_size = 2
    features = [
        torch.randn(batch_size, 144, 256, 256),   # Level 3 (highest resolution)
        torch.randn(batch_size, 288, 128, 128),  # Level 2
        torch.randn(batch_size, 576, 64, 64),    # Level 1
        torch.randn(batch_size, 1152, 32, 32),   # Level 0 (lowest resolution)
    ]

    # 2. Initialize position encoding
    pos_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        temperature=10000
    )

    # 3. Create MFBFpnNeck instance
    rfb_fpn = MFBFpnNeck(
        position_encoding=pos_encoding,
        d_model=256,  # Output channel dimension
        backbone_channel_list=[1152, 576, 288, 144],
        fpn_interp_model="bilinear",
        fuse_type="sum",
        fpn_top_down_levels=[2, 3]  # Only apply top-down pathway to last two levels
    )
    # fpn = FpnNeck(
    #     position_encoding=pos_encoding,
    #     d_model=256,  # Output channel dimension
    #     backbone_channel_list=[1152, 576, 288, 144],
    #     fpn_interp_model="bilinear",
    #     fuse_type="sum",
    #     fpn_top_down_levels=[2, 3]  # Only apply top-down pathway to last two levels
    # )

    # 4. Forward pass
    output_features, position_encodings = rfb_fpn(features)

    # 5. Print shapes of outputs
    print("Output Feature Shapes:")
    for i, feat in enumerate(output_features):
        print(f"Level {i}: {feat.shape}")

    print("\nPosition Encoding Shapes:")
    for i, pos in enumerate(position_encodings):
        print(f"Level {i}: {pos.shape}")

    # 6. Verify output dimensions
    expected_channels = 256
    for i, feat in enumerate(output_features):
        assert feat.shape[1] == expected_channels, f"Level {i} has wrong channel count"
        
    print("\nAll outputs have correct channel dimensions!")

def test_rfb_module():
    # 1. Create sample input
    batch_size = 2
    in_channels = 256  # 输入通道数
    out_channels = 128 # 输出通道数
    height = 64
    width = 64
    
    # 创建随机输入tensor
    x = torch.randn(batch_size, in_channels, height, width)

    # 2. Initialize MFB module
    mfb = MFB_modified(
        in_channel=in_channels,
        out_channel=out_channels
    )
    
    # 3. Forward pass
    output = mfb(x)
    
    # 4. Print shapes and verify
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 5. Verify branch outputs individually
    with torch.no_grad():
        # Test individual branches
        x0 = mfb.branch0(x)
        x1 = mfb.branch1(x)
        x2 = mfb.branch2(x)
        x3 = mfb.branch3(x)
        
        print("\nBranch output shapes:")
        print(f"Branch 0: {x0.shape}")
        print(f"Branch 1: {x1.shape}")
        print(f"Branch 2: {x2.shape}")
        print(f"Branch 3: {x3.shape}")
        
        # Test concatenation
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        print(f"\nConcatenated shape: {x_cat.shape}")
        
        # Test residual connection
        res = mfb.conv_res(x)
        print(f"Residual shape: {res.shape}")    

if __name__ == "__main__":
    # test_rfb_module()
    test_neck()
