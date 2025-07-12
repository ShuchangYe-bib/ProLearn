import torch
import torch.nn as nn
from swin_unet import SwinTransformerSys
from trans_unet import TransUNet
from basic_unets import U_Net, AttU_Net
from lvit import LViT
from unetpp import UNetPP


image = torch.randn(32, 3, 224, 224)

text = torch.randn(32, 10, 768)

swin_unet = SwinTransformerSys()

swin_unet_output = swin_unet(image)

print(swin_unet_output.shape)

trans_unet = TransUNet()

trans_unet_output = swin_unet(image)

print(trans_unet_output.shape)

unet = U_Net()

unet_output = unet(image)

print(unet_output.shape)

att_unet = AttU_Net()

att_unet_output = att_unet(image)

print(att_unet_output.shape)

lvit = LViT()

lvit_output = lvit(image, text)

print(lvit_output.shape)

unet_pp = UNetPP()

unet_pp_output = unet_pp(image)

print(unet_pp_output.shape)








