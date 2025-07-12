# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair
from einops import rearrange, repeat
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from transformers import BertTokenizer
import ml_collections


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase
# test_session = "Test_session_05.23_14h19"  # dice=79.98, IoU=66.83

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class LViT(nn.Module):
    def __init__(self, config=get_CTranS_config(), n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            max_position_embeddings=50,
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_embeddings = BertEmbeddings(self.bert_config)
        self.token_type_embeddings = nn.Embedding(2, 768)
        # # freeze the parameters
        # for param in self.text_embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        ## to reuse the dataset code for ariadne
        image, text = x
        text = self.tokenizer(text,
            padding="max_length", 
            truncation=True, 
            max_length=10,  # Match BertConfig's max_position_embeddings
            return_tensors="pt"
        )
        # print(text)
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        x = image
        text = self.text_embeddings(text["input_ids"].to(image.device))
        if text.shape[1] > 10: # this is forced by LViT
            text = text[:, :10, :]

        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        y1 = self.downVit(x1, x1, text1)
        x2 = self.down1(x1)
        y2 = self.downVit1(x2, y1, text2)
        x3 = self.down2(x2)
        y3 = self.downVit2(x3, y2, text3)
        x4 = self.down3(x3)
        y4 = self.downVit3(x4, y3, text4)
        x5 = self.down4(x4)
        y4 = self.upVit3(y4, y4, text4, True)
        y3 = self.upVit2(y3, y4, text3, True)
        y2 = self.upVit1(y2, y3, text2, True)
        y1 = self.upVit(y1, y2, text1, True)
        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits




class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Embeddings(nn.Module):
    # Construct the patch, position embeddings
    def __init__(self, config, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        # img_size[0]=img
        # patch_size[0]=patch_size[1] [16, 8, 4, 2]
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # [B, num_patches, hidden_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_patches, out_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, num_patches, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B, num_patches, 3, num_heads, per_HeadDim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_patches, per_HeadDim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, num_patches, per_HeadDim] [4, 8, 196, 8/16/32/64] easy to use tensor 

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)  # [B, num_heads, num_patches, per_HeadDim]
        x = x.transpose(1, 2)  # [B, num_patches, num_heads, per_HeadDim]
        x = x.reshape(B, N, C)  # [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, embed_dim]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class VisionTransformer(nn.Module):  # Transformer-branch
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.vis = vis
        self.embeddings = Embeddings(config=config, patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        act_layer = nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)
        self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)
        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)

    def forward(self, x, skip_x, text, reconstruct=False):
        if not reconstruct:
            x = self.embeddings(x)
            if self.dim == 64:
                x = x+self.CTBN3(text)  # [B, num_patches, embed_dim]
            x = self.Encoder_blocks(x)
        else:
            x = self.Encoder_blocks(x)
        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        elif not reconstruct:
            x = x.transpose(1, 2)  # [B, embed_dim, num_patches]
            x = self.CTBN(x)  # [B, embed_dim//2, num_patches]
            x = x.transpose(1, 2)  # [B, num_patches, embed_dim//2]
            y = torch.cat([x, skip_x], dim=2)  # [B, num_patches, embed_dim]
            return y
        elif reconstruct:
            skip_x = skip_x.transpose(1, 2)
            skip_x = self.CTBN2(skip_x)
            skip_x = skip_x.transpose(1, 2)
            y = x+skip_x
            return y

class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2 
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),  # 2, 2*self.
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    '''forward'''

    def forward(self, x):
        x_avg = self.conv_avg(x)  
        x_avg = self.relu_avg(x_avg) 
        x_avg = torch.mean(x_avg, dim=1)
        x_avg = x_avg.unsqueeze(dim=1)
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)
        x_out = x_max+x_avg
        x_output = torch.cat((x_avg, x_max, x_out), dim=1) 
        x_output = x_output.transpose(1, 3) 
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3) 
        y = x_output * x
        return y