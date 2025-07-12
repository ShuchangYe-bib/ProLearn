# === Standard Library ===
from typing import List, Optional

# === Third-party Libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import repeat
from tqdm import tqdm

# === BiomedCLIP Utilities ===
from open_clip import create_model_from_pretrained, get_tokenizer


class BiomedCLIP(nn.Module):
    """
    BiomedCLIP wrapper for image-text representation learning and contrastive pretraining.
    """

    def __init__(self, args):
        """
        Initialize BiomedCLIP model and tokenizer.

        Args:
            args (Namespace): Configuration object with model and training settings.
        """
        super().__init__()
        self.args = args
        self.temperature = 0.1
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        # Projection + Classification Head for Image-Text Matching
        feature_dim = 512
        self.itm_head = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2)
        )

    def encode_image(self, image: torch.Tensor, preprocess: bool = False) -> torch.Tensor:
        """
        Encodes image using visual encoder.

        Args:
            image (Tensor or str): Input tensor [B, C, H, W] or image path.
            preprocess (bool): Whether to apply preprocessing.

        Returns:
            Tensor: Encoded image features.
        """
        if preprocess:
            image = self.preprocess(Image.open(image)).unsqueeze(0)

        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b 3 h w')  # Convert grayscale to RGB

        return self.model.encode_image(image.to(self.args.device))

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Tokenizes and encodes text using BERT encoder.

        Args:
            text (List[str]): List of captions or textual descriptions.

        Returns:
            Tensor: Encoded text features.
        """
        tokens = self.tokenizer(text)
        return self.model.encode_text(tokens.to(self.args.device))

    def encode_image_feature(self, image: torch.Tensor, preprocess: bool = False) -> torch.Tensor:
        """
        Extracts intermediate visual features (before projection).

        Returns:
            Tensor: [B, N, C] features.
        """
        if preprocess:
            image = self.preprocess(Image.open(image)).unsqueeze(0)
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b 3 h w')
        return self.model.visual.forward_feature(image.to(self.args.device))

    def encode_text_feature(self, text: List[str]) -> torch.Tensor:
        """
        Extracts intermediate text features (BERT transformer hidden states).

        Returns:
            Tensor: [B, L, C] features.
        """
        tokens = self.tokenizer(text)
        return self.model.text_encoder.forward_feature(tokens.to(self.args.device))

    def contrastive_loss(
        self, image: torch.Tensor, text: List[str], pseudolabel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes contrastive and image-text matching (ITM) losses.

        Returns:
            Tensor: Combined loss value.
        """
        B = image.size(0)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # === Contrastive Loss ===
        logits = torch.matmul(image_features, text_features.T) / self.temperature

        if pseudolabel is not None:
            label_sim = (pseudolabel.unsqueeze(1) == pseudolabel.unsqueeze(0)).all(dim=-1).float()
            contrastive_targets = label_sim.to(logits.device)
        else:
            contrastive_targets = torch.arange(B, device=logits.device)

        loss_img = F.cross_entropy(logits, contrastive_targets)
        loss_txt = F.cross_entropy(logits.T, contrastive_targets)
        contrastive_loss = (loss_img + loss_txt) / 2

        # === Image-Text Matching (ITM) Loss ===
        pos_pairs = torch.cat([image_features, text_features], dim=-1)
        neg_text = text_features[torch.randperm(B)]
        neg_pairs = torch.cat([image_features, neg_text], dim=-1)

        itm_input = torch.cat([pos_pairs, neg_pairs], dim=0)  # [2B, 2D]
        itm_labels = torch.cat([
            torch.ones(B, dtype=torch.long),
            torch.zeros(B, dtype=torch.long)
        ]).to(image.device)

        itm_logits = self.itm_head(itm_input)
        itm_loss = F.cross_entropy(itm_logits, itm_labels)

        return contrastive_loss + itm_loss

    def contrastive_learning(self, data, epochs: int = 5, lr: float = 1e-4):
        """
        Performs contrastive pretraining with BiomedCLIP.

        Args:
            data (DataLoader): Iterable with dicts: {image, text, pseudolabel}.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = AdamW(
            list(self.model.visual.head.parameters()) +
            list(self.model.text.proj.parameters()),
            lr=lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                image = batch["image"]
                text = batch["text"]
                pseudolabel = batch["pseudolabel"]["position"]

                optimizer.zero_grad()
                loss = self.contrastive_loss(image, text, pseudolabel=pseudolabel)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            print(f"[Epoch {epoch+1}] Contrastive Pretrain Loss: {total_loss / len(data):.4f}")
