# === Third-party Libraries ===
import torch
import torch.nn as nn
from einops import rearrange, repeat
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from monai.transforms import Compose, ToTensord
from transformers import AutoTokenizer

# === Project Modules ===
from .layers import GuideDecoder, GuidedApproximation
from .networks import BERTModel, CLIPBERTModel, VisionModel


class MMUNet(nn.Module):
    """
    Multimodal UNet (MMUNet): Combines image and text features via prototype-based guidance and attention decoding.
    """

    def __init__(self, args, prototype=None):
        """
        Initialize MMUNet.

        Args:
            args (Namespace): Configuration arguments.
            prototype (Optional[nn.Module]): Prototype module used for semantic guidance.
        """
        super(MMUNet, self).__init__()

        # === Backbone Encoders ===
        self.vision_encoder = VisionModel(args.vision_type)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)
        self.text_encoder = CLIPBERTModel(args)
        self.prototype = prototype

        # === Decoder Configuration ===
        self.spatial_dim = [7, 14, 28, 56]  # Spatial resolutions for skip connections
        feature_dim = [768, 384, 192, 96]   # Corresponding channel dimensions

        if args.agg == "sum":
            self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24, args.text_len, args.project_dim)
            self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12, args.text_len, args.project_dim)
            self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9, args.text_len, args.project_dim)

        elif args.agg in {"linear", "attention"}:
            self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], args.num_candidate, args.num_candidate, args.text_dim)
            self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], args.num_candidate, args.num_candidate, args.text_dim)
            self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], args.num_candidate, args.num_candidate, args.text_dim)

            if args.agg == "attention":
                self.approx1 = GuidedApproximation(args.text_dim, args.num_candidate, feature_dim[0], feature_dim[1])
                self.approx2 = GuidedApproximation(args.text_dim, args.num_candidate, feature_dim[1], feature_dim[2])
                self.approx3 = GuidedApproximation(args.text_dim, args.num_candidate, feature_dim[2], feature_dim[3])

        # Final upsampling and segmentation head
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def tokenize(self, captions, device):
        """
        Tokenize a list of text captions.

        Args:
            captions (List[str]): List of textual descriptions.
            device (torch.device): Device to move token tensors to.

        Returns:
            dict: Tokenized inputs containing 'input_ids' and 'attention_mask'.
        """
        input_ids, attention_masks = [], []
        for caption in captions:
            encoding = self.tokenizer.encode_plus(
                caption,
                padding='max_length',
                max_length=24,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoding['input_ids'].squeeze(0).tolist())
            attention_masks.append(encoding['attention_mask'].squeeze(0).tolist())

        tensor_input = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks)
        }

        # Convert to tensors and move to device
        tensor_input = Compose([ToTensord(["input_ids", "attention_mask"])])(tensor_input)
        return {
            'input_ids': tensor_input['input_ids'].to(device),
            'attention_mask': tensor_input['attention_mask'].to(device)
        }

    def pseg(self, image, image_emb):
        """
        Prototype-guided segmentation inference.

        Args:
            image (Tensor): Input image tensor.
            image_emb (Tensor): Image-level embedding.

        Returns:
            Tensor: Segmentation prediction (B, 1, H, W)
        """
        image_features = self.vision_encoder(image)

        # Handle feature extraction for ConvNeXt/ViT
        if image_features[0].dim() == 4:
            image_features = image_features[1:]
            image_features = [rearrange(fmap, 'b c h w -> b (h w) c') for fmap in image_features]

        os32 = image_features[3]
        ref0 = self.prototype.query(image, image_emb)
        ref1 = self.approx1(ref0, os32, image_features[2])
        os16 = self.decoder16(os32, image_features[2], ref1)

        ref0 = self.prototype.query(image, image_emb)
        ref2 = self.approx2(ref0, os16, image_features[1])
        os8 = self.decoder8(os16, image_features[1], ref2)

        ref0 = self.prototype.query(image, image_emb)
        ref3 = self.approx3(ref0, os8, image_features[0])
        os4 = self.decoder4(os8, image_features[0], ref3)

        # Reshape and decode to segmentation map
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        seg = self.out(os1).sigmoid()

        return seg

    def forward(self, data):
        """
        Forward pass for inference.

        Args:
            data (dict): Input dictionary with keys: 'image', 'image_emb'

        Returns:
            Tensor: Segmentation logits
        """
        image = data["image"]
        image_emb = data["image_emb"]

        # Convert grayscale to RGB
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)

        seg = self.pseg(image=image, image_emb=image_emb)
        return seg
