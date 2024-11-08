import argparse
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from datasets_test import data_loader_test
from datasets_test import convert_to_hake_format
from datasets_test import convert_indices_to_words
import util.misc as utils
from datasets import build_dataset
from datasets import hake_meta

# 跟踪输出
# import sys
# import traceback
#
# old_f = sys.stdout
# class F:
#     def write(self, x):
#         old_f.write(x.replace("\n", " [%s]\n" % str(traceback.extract_stack())))
# sys.stdout = F()

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--arch', default='ViT-B/16', type=str)
    parser.add_argument('--clip_backbone_pretrained', default="pretrained/clip/ViT-B-16.new.pt", type=str)
    parser.add_argument('--backbone_freeze_layer', default=12, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)   #default=1e-3
    parser.add_argument('--weight_decay', default=1e-4, type=float)  #default=1e-5
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--resume', default='runs/train/exp75/checkpoint_epoch_13_avgloss_1115.7449.pth', help='resume from checkpoint')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--vis_test', default=False, action='store_true')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./runs/',
                        help='path where to save, empty for no saving')
    return parser




class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


# class Unflatten(nn.Module):
#     def __init__(self, start_axis, shape):
#         super(Unflatten, self).__init__()
#         self.start_axis = start_axis
#         self.shape = shape
#
#     def forward(self, x):
#         return x.view(*x.shape[:self.start_axis], *self.shape)


class Unflatten(nn.Module):
    def __init__(self, start_axis, shape):
        super(Unflatten, self).__init__()
        self.start_axis = start_axis
        self.shape = shape

    def forward(self, x):
        # 打印输入张量的形状
        print(f"Input shape before unflatten: {x.shape}")
        print(f"start_axis: {self.start_axis}")
        print(f"shape: {self.shape}")

        input_elements = x.numel()  # 输入张量的元素总数
        output_elements = np.prod(x.shape[:self.start_axis]) * np.prod(self.shape)  # 目标形状的元素总数

        print(f"Input elements: {input_elements}")
        print(f"Output elements: {output_elements}")

        if input_elements != output_elements:
            raise ValueError("Total number of elements must be the same for input and target shapes")

        return x.view(*x.shape[:self.start_axis], *self.shape)



class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])

        #print("(x[:, self.start_index:]).shape",(x[:, self.start_index:]).shape)
        features = torch.cat((x[:, self.start_index:], readout), dim=-1)
        return self.project(features)


from transformers import CLIPModel, CLIPVisionConfig
import torch.nn as nn

class ViT_Origin(VisionTransformer):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer, **kwargs)
        self.patch_size = patch_size
        self.start_index = 1
        features = [256, 512, 1024, 1024]
        readout_oper = [
            ProjectReadout(embed_dim, self.start_index) for _ in features
        ]
        self.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),

            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.norm = nn.Identity()
        self.head = nn.Identity()

    def _resize_pos_embed(self, posemb, gs_h, gs_w):
        posemb_tok, posemb_grid = (
            posemb[:, : self.start_index],
            posemb[0, self.start_index:],
        )

        gs_old = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(
            (1, gs_old, gs_old, -1)).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(
            posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(
            0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward(self, x):
        b, c, h, w = x.shape

        pos_embed = self._resize_pos_embed(
            self.pos_embed, h // self.patch_size, w // self.patch_size
        )
        x = self.patch_embed.proj(x).flatten(2).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        outputs = []
        for index, blk in enumerate(self.blocks):
            x = blk(x)
            if index in [5, 11, 17, 23]:
                outputs.append(x)
        print("outputs",outputs[0].shape)
        layer_1 = self.act_postprocess1[0:2](outputs[0])
        layer_2 = self.act_postprocess2[0:2](outputs[1])
        layer_3 = self.act_postprocess3[0:2](outputs[2])
        layer_4 = self.act_postprocess4[0:2](outputs[3])

        shape = (-1, 1024, h // self.patch_size, w // self.patch_size)
        layer_1 = layer_1.view(shape)
        layer_2 = layer_2.view(shape)
        layer_3 = layer_3.view(shape)
        layer_4 = layer_4.view(shape)
        # print(layer_1.shape)
        # print(layer_2.shape)
        # print(layer_3.shape)
        # print(layer_4.shape)
        layer_1 = self.act_postprocess1[3:](layer_1)
        layer_2 = self.act_postprocess2[3:](layer_2)
        layer_3 = self.act_postprocess3[3:](layer_3)   #2
        layer_4 = self.act_postprocess4[3:](layer_4)   #2

        return layer_1, layer_2, layer_3, layer_4


from transformers import CLIPVisionModel
import torch.nn.functional as F
import torch.nn as nn
import math

class ViT(VisionTransformer):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=1024, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
                 norm_layer=nn.LayerNorm, **kwargs):
        super(ViT, self).__init__()

        config = CLIPVisionConfig(
            image_size=384,  # 图像大小
            patch_size=16,  # 图像块大小
            num_channels=3,  # 图像通道数
            hidden_size=1024,  # 嵌入维度
            num_hidden_layers=12,  # 隐藏层数量
            num_attention_heads=12,  # 注意力头数量
            intermediate_size=1024 * 4,  # 中间层大小，通常是 embed_dim * mlp_ratio
            hidden_act="gelu",  # 激活函数
            hidden_dropout_prob=0.0,  # 隐藏层 dropout 概率
            attention_probs_dropout_prob=0.0,  # 注意力 dropout 概率
            initializer_range=0.02,  # 初始化范围
            layer_norm_eps=1e-5,  # 层归一化 epsilon
            qkv_bias=True,  # 是否使用 qkv 偏置
            use_cache=True,  # 是否使用缓存
        )
        # 加载预训练的clip-vit-base-patch16模型
        self.clip_vit = CLIPVisionModel.from_pretrained('./clip-vit-base-patch16')

        self.patch_size = patch_size
        self.start_index = 1
        features = [256, 512, 1024, 1024]  # 特征通道数与原始模型保持一致
        self.dropout = nn.Dropout(p=drop_rate)  # 添加dropout层
        # 处理每个特征层的输出
        readout_oper = [
            ProjectReadout(embed_dim, self.start_index) for _ in features
        ]

        self.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[0], kernel_size=1),
            nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4)
        )

        self.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[1], kernel_size=1),
            nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2)
        )

        self.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[2], kernel_size=1)
        )

        self.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(in_channels=embed_dim, out_channels=features[3], kernel_size=1),
            nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1)
        )

        self.norm = nn.Identity()
        self.head = nn.Identity()
        self.linear_layer = nn.Linear(768, 1024).to("cuda")

    def _resize_pos_embed(self, posemb, gs_h, gs_w):
        # CLIP 的位置嵌入是 nn.Embedding，需要转换为权重张量
        posemb_weights = posemb.weight  # 提取 nn.Embedding 中的权重

        # 从位置嵌入中分离 CLS token 和网格嵌入
        posemb_tok, posemb_grid = (
            posemb_weights[:self.start_index],
            posemb_weights[self.start_index:],
        )

        gs_old = int(math.sqrt(len(posemb_grid)))

        # 调整网格嵌入的形状并插值到新的网格大小
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        # 重新组合 CLS token 和插值后的网格嵌入
        posemb = torch.cat([posemb_tok.unsqueeze(0), posemb_grid], dim=1)

        return posemb

    def forward(self, x):
        b, c, h, w = x.shape

        # 正确获取 CLIP 模型的位置信息嵌入
        pos_embed = self._resize_pos_embed(
            self.clip_vit.vision_model.embeddings.position_embedding,  # 修正访问路径
            h // self.patch_size, w // self.patch_size
        )

        # 获取 patch 嵌入，使用 CLIP 的 patch_embedding 层代替
        x = self.clip_vit.vision_model.embeddings.patch_embedding(x).flatten(2).permute(0, 2, 1)

        cls_tokens = self.clip_vit.vision_model.embeddings.class_embedding
        cls_tokens = cls_tokens.unsqueeze(0).repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = self.dropout(x)

        outputs = []

        #         for index, blk in enumerate(self.blocks):
        #             x = blk(x)
        #             if index in [5, 11, 17, 23]:
        #                 outputs.append(x)



        for index, blk in enumerate(self.blocks):
            x = blk(x)
            if index in [3, 6, 9, 11]:  # 提取不同层的输出3, 6, 9, 11
                outputs.append(x)
        # 应用全连接层
        outputs[0] = self.linear_layer(outputs[0])
        outputs[1] = self.linear_layer(outputs[1])
        outputs[2] = self.linear_layer(outputs[2])
        outputs[3] = self.linear_layer(outputs[3])
        layer_1 = self.act_postprocess1[0:2](outputs[0])
        layer_2 = self.act_postprocess2[0:2](outputs[1])
        layer_3 = self.act_postprocess3[0:2](outputs[2])
        layer_4 = self.act_postprocess4[0:2](outputs[3])

        shape = (-1, 1024, h // self.patch_size, w // self.patch_size)
        layer_1 = layer_1.view(shape)
        layer_2 = layer_2.view(shape)
        layer_3 = layer_3.view(shape)
        layer_4 = layer_4.view(shape)

        layer_1 = self.act_postprocess1[3:](layer_1)
        layer_2 = self.act_postprocess2[3:](layer_2)
        layer_3 = self.act_postprocess3[3:](layer_3)
        layer_4 = self.act_postprocess4[3:](layer_4)

        return layer_1, layer_2, layer_3, layer_4

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

# class CLIPText(nn.Module):
#     def __init__(
#             self,
#             max_text_length: int = 77,
#             vocab_size: int = 49408,
#             text_embed_dim: int = 512,
#             text_heads: int = 8,
#             text_layers: int = 12,
#             text_hidden_act: str = "gelu",
#             projection_dim: int = 512,
#             local_model_path: str = "./clip-vit-base-patch32/"):
#         super().__init__()
#
#         # Load the CLIP text model from local directory
#         self.text_model = CLIPTextModel.from_pretrained(local_model_path)
#         self.text_projection = nn.Parameter(torch.randn(text_embed_dim, projection_dim))

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPConfig
from transformers import CLIPModel, CLIPConfig

class CLIPText(nn.Module):
    def __init__(
            self,
            max_text_length: int = 77,
            vocab_size: int = 49408,
            text_embed_dim: int = 512,
            text_heads: int = 8,
            text_layers: int = 12,
            text_hidden_act: str = "gelu",
            projection_dim: int = 512):
        super().__init__()

        # Define custom configuration
        config = CLIPConfig(
            vocab_size=vocab_size,
            hidden_size=text_embed_dim,
            num_attention_heads=text_heads,
            num_hidden_layers=text_layers,
            intermediate_size=text_embed_dim * 4,
            hidden_act=text_hidden_act,
            max_position_embeddings=max_text_length,
            attention_dropout = 0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            torchscript=False,
            use_cache=True,
            tie_word_embeddings=False,
            projection_dim=projection_dim,
            eos_token_id=49407

        )

        # Instantiate text model with custom configuration
        self.text_model = CLIPTextModel(config)
        self.text_model = CLIPTextModel.from_pretrained("./clip-vit-base-patch16")
        # Projection layer
        self.text_projection = nn.Parameter(torch.randn(text_embed_dim, projection_dim))


# class CLIPText(nn.Module):
#     def __init__(
#             self,
#             local_weights_path: str = "pretrained/ViT-B-16.new.pt",
#             max_text_length: int = 77,
#             text_embed_dim: int = 512,
#             projection_dim: int = 512):
#         super().__init__()
#
#         # Manually define the CLIP configuration
#         config = CLIPConfig(
#             hidden_size=768,  # Corresponds to the hidden size of ViT-B/16
#             num_attention_heads=12,  # Number of heads in ViT-B/16
#             num_hidden_layers=12,  # Number of transformer layers in ViT-B/16
#             intermediate_size=3072,  # Intermediate size for MLP layers
#             projection_dim=512,  # Projection dimension used by CLIP
#             max_position_embeddings=197,  # 196 patches + 1 CLS token for ViT-B/16
#             vocab_size=49408  # Text model's vocabulary size
#         )
#
#         # Initialize the CLIP model without downloading weights
#         self.clip_model = CLIPModel(config)
#
#         # Load the local ViT weights into the vision model
#         state_dict = torch.load(local_weights_path)
#         self.clip_model.load_state_dict(state_dict)
#
#         # Extract text and vision models from the loaded CLIP model
#         self.text_model = self.clip_model.text_model
#         self.text_projection = nn.Parameter(torch.randn(text_embed_dim, projection_dim))


    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        # print(text_outputs,'text_outpuyts')
        # pooled_output = text_outputs[1] if return_dict else text_outputs.last_hidden_state[:, 0, :]
        # text_features = torch.matmul(pooled_output, self.text_projection)
        # return text_features

        pooled_output = text_outputs[1]
        text_features = torch.matmul(pooled_output, self.text_projection)
        return text_features





import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


class ResidualConvUnitCustom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlockCustom(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation=nn.ReLU(), deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlockCustom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features if not expand else features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnitCustom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnitCustom(features, activation, bn)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class DifferentiableArgmax(nn.Module):
    def __init__(self, temperature=0.1):
        super(DifferentiableArgmax, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)



class Scratch(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 1024], out_channels=256):
        super().__init__()
        self.out_c = 512
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.layer1_rn = nn.Conv2d(in_channels[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_channels[1], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_channels[2], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_channels[3], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.refinenet2 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.refinenet3 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.refinenet4 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.head1 = nn.Conv2d(out_channels, self.out_c, kernel_size=1)
        self.output_conv = nn.Sequential(Interpolate(scale_factor=2, mode="bilinear", align_corners=True))
        # 添加一个线性层，用于将 logits_per_image 转换为 [5, 512] 的形状

        self.fc = nn.Linear(50176, 512)
        # 添加两个全连接层
        self.fc1 = nn.Linear(50176, 10240)
        self.fc2 = nn.Linear(10240, 1024)
        self.fc3 = nn.Linear(1024, 512)
        # 添加一个 GumbelSoftmax 层


    def forward(self, layer_1, layer_2, layer_3, layer_4, text_features):
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        image_features = self.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
       # print("image_features", image_features.shape)
        #print("text_features",text_features.shape)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * torch.matmul(image_features, text_features.t())  #相似度差异
        # print("logits_per_ori_image", logits_per_image.shape)
        # 转置 logits_per_image 的形状，从 [102400, 5] 到 [5, 102400]
        out2 = logits_per_image.t()
        out2 = self.fc(out2)
        out2 = out2.sum(dim=-1, keepdim=True)
        # 使用线性层转换 logits_per_image 的形状，从 [5, 102400] 到 [5, 512]
        # out2 = self.fc1(out2)
        # out2 = self.fc2(out2)
        # out2 = self.fc3(out2)

        #
        # out = logits_per_image.reshape(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        # # print("out_ori", out.shape)
        # out = self.output_conv(out)    #插值恢复原始大小
        # # print(f'out type: {out.dtype}, ori_permute_shape: {out.shape}')
        return out2


class LSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPText()
        self.vit = ViT()
        self.scratch = Scratch()

    def forward(self, images, texts):
        #分支1
        layer_1, layer_2, layer_3, layer_4 = self.vit.forward(images)
        text_features = self.clip.get_text_features(texts)

        text_features2 = self.clip.get_text_features(texts)
        text_features3 = self.clip.get_text_features(texts)

        return self.scratch.forward(layer_1, layer_2, layer_3, layer_4, text_features)




import torchvision.transforms as transforms
import numpy as np
from transformers import CLIPTokenizer

# 使用 CLIPTokenizer 对标签进行编码
tokenizer =  CLIPTokenizer.from_pretrained("./clip-vit-base-patch16/")
#text_model = CLIPTextModel.from_pretrained("./clip-vit-base-patch32/")
eos_token_id = tokenizer.eos_token_id

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import xml.etree.ElementTree as ET

# # Define action categories
# ACTIONS = [
#     "applauding",
#     "blowing_bubbles",
#     "brushing_teeth",
#     "cleaning_the_floor",
#     "climbing",
#     "cooking",
#     "cutting_trees",
#     "cutting_vegetables",
#     "drinking",
#     "feeding_a_horse",
#     "fishing",
#     "fixing_a_bike",
#     "fixing_a_car",
#     "gardening",
#     "holding_an_umbrella",
#     "jumping",
#     "looking_through_a_microscope",
#     "looking_through_a_telescope",
#     "playing_guitar",
#     "playing_violin",
#     "pouring_liquid",
#     "pushing_a_cart",
#     "reading",
#     "phoning",
#     "riding_a_bike",
#     "riding_a_horse",
#     "rowing_a_boat",
#     "running",
#     "shooting_an_arrow",
#     "smoking",
#     "taking_photos",
#     "texting_message",
#     "throwing_frisby",
#     "using_a_computer",
#     "walking_the_dog",
#     "washing_dishes",
#     "watching_TV",
#     "waving_hands",
#     "writing_on_a_board",
#     "writing_on_a_book"
# ]
#
#
# # 自定义数据集类
# class Stanford40Dataset(Dataset):
#     def __init__(self, root_dir, annotations_dir, transform=None):
#         self.root_dir = root_dir
#         self.annotations_dir = annotations_dir
#         self.transform = transform
#         self.image_paths, self.labels = self.load_annotations()
#
#     def load_annotations(self):
#         image_paths = []
#         labels = []
#         for filename in os.listdir(self.annotations_dir):
#             if not filename.endswith('.xml'):
#                 continue
#             xml_path = os.path.join(self.annotations_dir, filename)
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#
#             image_file = root.find('filename').text
#             action = root.find('.//action').text
#
#             image_paths.append(os.path.join(self.root_dir, image_file))
#             labels.append(ACTIONS.index(action))
#
#         return image_paths, labels
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, ACTIONS[label],label

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),   #960, 640     #320 320
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# custom loss functions
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, model_output, text_embedding):
        return nn.functional.mse_loss(model_output, text_embedding)


import os


def save_checkpoint(state, run_dir, epoch, avg_loss):
    # Format the average loss to four decimal places
    avg_loss_str = f'{avg_loss:.4f}'

    # Create the checkpoint file path with epoch number and average loss
    checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch + 1}_avgloss_{avg_loss_str}.pth')

    # Save the checkpoint
    torch.save(state, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')

def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler):
    if os.path.exists(checkpoint_path):
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        # bug fix, Prevent overwriting lr...
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        args = checkpoint['args']
        print(f'Checkpoint loaded: {checkpoint_path}')
        return start_epoch, args
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")



def train(model, dataloader, criterion, optimizer, args):

    if   args.resume:
        checkpoint_path = args.resume
        args.start_epoch, args = load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        print(f"Starting from epoch: {args.start_epoch}")
    # Output information about the optimizer, lr_scheduler, and loaded model
    print("Optimizer details:", optimizer)
    #print("Learning Rate Scheduler details:", lr_scheduler)
    #print("Model loaded:", model)



    # Create a unique directory for each run
    run_number = 1
    run_dir = f'{args.output_dir}/train/exp'

    while os.path.exists(run_dir):
        run_number += 1
        run_dir = f'{args.output_dir}/train/exp{run_number}'

    os.makedirs(run_dir, exist_ok=True)




    for epoch in range(args.start_epoch,args.epochs + args.start_epoch):
        model.train()
        running_loss = 0.0

        log_file = run_dir + '/training_log.txt'
        save_interval = 100
        metric_logger = utils.MetricLogger(log_file,save_interval,delimiter="  ")
        header = 'Train:'


        for img, gt_labels, img_metas in metric_logger.log_every(dataloader, 100, header):
            img_meta_dict = img_metas[0]
            gt_to_text = convert_to_hake_format(gt_labels, img_meta_dict)
            gt_to_words = convert_indices_to_words(gt_to_text, hake_meta.class_names)


            images = img.to(args.device)
            label = torch.tensor(gt_to_text['gt_verbs']).to(args.device)
            label_gt_pasta_foot = torch.tensor(gt_to_text['gt_pasta_foot']).to(args.device)
            label_gt_pasta_leg = torch.tensor(gt_to_text['gt_pasta_leg']).to(args.device)


            texts = tokenizer(gt_to_words['gt_verbs'], add_special_tokens=True,padding=True, truncation=True, return_tensors="pt")['input_ids']
            texts = texts.to(args.device, dtype=torch.int32)
            texts_gt_pasta_foot = tokenizer(gt_to_words['gt_pasta_foot'], add_special_tokens=True, padding=True, truncation=True,
                              return_tensors="pt")['input_ids']
            texts_pasta = texts_gt_pasta_foot.to(args.device, dtype=torch.int32)
            texts_gt_pasta_leg = tokenizer(gt_to_words['gt_pasta_leg'], add_special_tokens=True, padding=True, truncation=True,
                              return_tensors="pt")['input_ids']
            texts_pasta_binary = texts_gt_pasta_leg.to(args.device, dtype=torch.int32)

            # print("gt_to_words['gt_verbs']",gt_to_words['gt_verbs'])
            # print("texts.shape:",texts.shape)
            optimizer.zero_grad()

            # Forward pass

            #修改model为三分支，输出为三个分支
            output = model(images, texts)
            # print("output,label",output,label)
            label = label.unsqueeze(1)
            loss = criterion(output.float(), label.float())



            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()

            # Update MetricLogger with the current loss
            metric_logger.update(loss=loss.item())


        avg_loss = running_loss / len(dataloader)
        metric_logger.update(loss=avg_loss)
        metric_logger.save_to_log()

        # Save checkpoint after each epoch
        save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, run_dir, epoch, avg_loss)



if __name__ == '__main__':
    # Initialize models, loss functions, and optimizers
    parser = argparse.ArgumentParser('LgAR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model = LSeg()
    model.to(args.device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataloader = data_loader_test(args)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)*args.epochs, eta_min=1e-6)
    train(model, dataloader, criterion, optimizer, args)

    # # Stanford40
    # root_dir = 'D:/Y/Stanford40/JPEGImages'
    # annotations_dir = 'D:/Y/Stanford40/XMLAnnotations'
    # dataset = Stanford40Dataset(root_dir=root_dir, annotations_dir=annotations_dir, transform=transform)
    #  random
    # num_samples = 1000
    # indices = np.random.choice(len(dataset), num_samples, replace=False)
    # from torch.utils.data.sampler import SubsetRandomSampler
    # sampler = SubsetRandomSampler(indices)
    # Stanford40dataloader = DataLoader(dataset, batch_size=4, sample