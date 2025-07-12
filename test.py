# === Standard Library ===
import argparse
import warnings

# === Third-party Libraries ===
import torch

# === Project Modules ===
from models import ProLearn, Prototype
from modules.dataset import MMSegDataset
from modules.dataloader import MMSegDataLoader
from modules.trainer import Trainer
import utils.config as config

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")


def get_parser():
    """
    Set up argument parser for model testing configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Language-Guided Medical Image Segmentation Inference")
    parser.add_argument(
        '--config', 
        default='./config/training.yaml', 
        type=str, 
        help='Path to configuration YAML file.'
    )
    parser.add_argument(
        '--v', 
        default='', 
        type=str, 
        help='Checkpoint version suffix (e.g., 1, 2, etc.).'
    )
    parser.add_argument(
        '--num_prototypes', 
        type=int, 
        default=None, 
        help='Number of prototypes per pseudo-label (optional).'
    )
    parser.add_argument(
        '--num_candidate', 
        type=int, 
        default=None, 
        help='Number of candidates used in response generation (optional).'
    )
    args = parser.parse_args()
    assert args.config is not None, "Configuration file must be provided."
    return args


def main():
    """
    Main function for loading model and performing inference on test set.
    """
    # Load configuration and override with CLI arguments
    args = get_parser()
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.num_prototypes:
        cfg.num_prototypes = args.num_prototypes
    if args.num_candidate:
        cfg.num_candidate = args.num_candidate

    # === Initialize prototype encoder and load precomputed features ===
    prototype = Prototype(cfg).to(cfg.device)
    prototype.load()

    # === Load trained segmentation model ===
    model = ProLearn(cfg, prototype).to(cfg.device)
    ckpt_path = f"{cfg.model_save_path}/{cfg.model_save_filename}"
    if args.v:
        ckpt_path += f"-v{args.v}"
    ckpt_path += ".ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)

    # === Prepare test dataset and dataloader ===
    ds_test = MMSegDataset(
        ann_path=cfg.ann_path,
        root_path=cfg.root_path,
        tokenizer=cfg.bert_type,
        image_size=cfg.image_size,
        mode='test',
        lazy=cfg.lazy
    )
    ds_test.precompute(encoder=prototype.encoder, device=cfg.device)

    dl_test = MMSegDataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # === Initialize trainer and run inference ===
    trainer = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        early_stopping_patience=None,
        train_loader=None,
        val_loader=None,
        test_loader=dl_test,
        model_save_path=None,
        model_name=None,
        max_epochs=None,
        device=cfg.device
    )
    trainer.test()


if __name__ == "__main__":
    main()


