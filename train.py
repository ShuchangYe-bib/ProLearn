# === Standard Library ===
import argparse
import random
import warnings

# === Third-party Libraries ===
import torch
import numpy as np
from torch.utils.data import random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# === Project Modules ===
import utils.config as config
from modules.dataset import MMSegDataset
from modules.dataloader import MMSegDataLoader
from modules.trainer import Trainer
from models import ProLearn, PSA

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")


def set_random_seed(seed: int):
    """
    Ensures deterministic training for reproducibility.

    Args:
        seed (int): Random seed to be set across all libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Language-Guided Medical Image Segmentation")
    parser.add_argument(
        '--config', 
        default='./config/training.yaml', 
        type=str, 
        help='Path to training configuration file.'
    )
    parser.add_argument(
        '--semi_ratio', 
        type=float, 
        default=1.0, 
        help='Ratio of training data to use for semi-supervised learning.'
    )
    parser.add_argument(
        '--num_prototypes', 
        type=int, 
        default=None, 
        help='Number of prototypes per pseudo label (optional).'
    )
    parser.add_argument(
        '--num_candidate', 
        type=int, 
        default=None, 
        help='Number of candidate responses to consider (optional).'
    )
    args = parser.parse_args()
    assert args.config is not None, "A config file path must be specified."
    return args


def main():
    """
    Main training pipeline for language-guided medical image segmentation.
    """
    # Load config and set seeds
    args = get_parser()
    cfg = config.load_cfg_from_cfg_file(args.config)
    set_random_seed(cfg.seed)

    # Update configuration with optional arguments
    if args.num_prototypes:
        cfg.num_prototypes = args.num_prototypes
    if args.num_candidate:
        cfg.num_candidate = args.num_candidate

    # Initialize prototype semantic alignment module (PSA)
    prototype = PSA(cfg).to(cfg.device)

    # Initialize model and optimization components
    model = ProLearn(cfg, prototype=prototype).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    # === Dataset Preparation ===
    ds_train = MMSegDataset(
        ann_path=cfg.ann_path,
        root_path=cfg.root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        aug=True,
        mode='train',
        lazy=cfg.lazy
    )

    # Semi-supervised data split
    ds_train_semi, _ = random_split(
        ds_train, 
        [int(args.semi_ratio * len(ds_train)), len(ds_train) - int(args.semi_ratio * len(ds_train))],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    ds_val = MMSegDataset(
        ann_path=cfg.ann_path,
        root_path=cfg.root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        mode='val',
        lazy=cfg.lazy
    )

    # === DataLoader Setup ===
    dl_train = MMSegDataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_train_semi = MMSegDataLoader(ds_train_semi, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_val = MMSegDataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # === Trainer Configuration ===
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=cfg.patience,
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=None,
        model_save_path=cfg.model_save_path,
        model_name=cfg.model_save_filename,
        max_epochs=cfg.max_epochs,
        device=cfg.device
    )

    # === Prototype Fitting and Feature Precomputation ===
    prototype.fit(dl_train_semi)

    ds_train.precompute(
        encoder=prototype.encoder,
        device=cfg.device
    )
    ds_val.precompute(
        encoder=prototype.encoder,
        device=cfg.device
    )

    # === Start Training ===
    trainer.train()


if __name__ == "__main__":
    main()
