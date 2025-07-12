import os
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, scheduler, early_stopping_patience, 
                 train_loader, val_loader, test_loader, model_save_path, model_name, 
                 max_epochs, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.device = device
        
        # Determine a unique filename for the checkpoint only once
        self.model_filename = None
        
    def _get_unique_model_filename(self, filename):
        version = 0
        unique_filename = f"{filename}.ckpt"
        while os.path.exists(os.path.join(self.model_save_path, unique_filename)):
            version += 1
            unique_filename = f"{filename}-v{version}.ckpt"
        return unique_filename

    def move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {key: self.move_to_device(value) for key, value in batch.items()}
        elif isinstance(batch, list):
            return [self.move_to_device(item) for item in batch]
        elif isinstance(batch, tuple):
            return tuple(self.move_to_device(item) for item in batch)
        return batch

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # x, y = batch
            # x, y = (x[0].to(self.device), x[1]), y.to(self.device)

            self.optimizer.zero_grad()
            # loss, _ = self.model(x, y)  # Forward pass with loss computation
            batch = self.move_to_device(batch)
            output = self.model(batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        acc_scores = []
        dice_scores = []
        miou_scores = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # x, y = batch
                # x, y = (x[0].to(self.device), x[1]), y.to(self.device)
                batch = self.move_to_device(batch)
                output = self.model(batch)
                preds, loss = output["logits"], output["loss"]
                # loss, preds = self.model(x, y)
                total_loss += loss.item()
                
                # Calculate scores
                acc_score = self.model.val_metrics["acc"](preds, output["label"]).item()
                acc_scores.append(acc_score)
                dice_score = self.model.val_metrics["dice"](preds, output["label"]).item()
                dice_scores.append(dice_score)
                miou_score = self.model.val_metrics["miou"](preds, output["label"]).item()
                miou_scores.append(miou_score)
                
        avg_acc = sum(acc_scores) / len(acc_scores)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_miou = sum(miou_scores) / len(miou_scores)
        return total_loss / len(self.val_loader), avg_acc, avg_dice, avg_miou

    def test_one_epoch(self):
        self.model.eval()
        total_loss = 0
        acc_scores = []
        dice_scores = []
        miou_scores = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                # x, y = batch
                # x, y = (x[0].to(self.device), x[1]), y.to(self.device)
                batch = self.move_to_device(batch)
                output = self.model(batch)
                preds, loss = output["logits"], output["loss"]
                # loss, preds = self.model(x, y)
                total_loss += loss.item()
                
                # Calculate dice score
                acc_score = self.model.test_metrics["acc"](preds, output["label"]).item()
                acc_scores.append(acc_score)
                dice_score = self.model.test_metrics["dice"](preds, output["label"]).item()
                dice_scores.append(dice_score)
                miou_score = self.model.test_metrics["miou"](preds, output["label"]).item()
                miou_scores.append(miou_score)
                
        avg_acc = sum(acc_scores) / len(acc_scores)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_miou = sum(miou_scores) / len(miou_scores)
        return total_loss / len(self.test_loader), avg_acc, avg_dice, avg_miou

    def save_checkpoint(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        if not self.model_filename:
            self.model_filename = self._get_unique_model_filename(self.model_name)
        model_path = os.path.join(self.model_save_path, self.model_filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    def train(self):
        best_val_dice = 0
        patience_counter = 0

        for epoch in range(self.max_epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_acc, val_dice, val_miou = self.validate_one_epoch()
            
            print(f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}, Val MIOU: {val_miou:.4f}")
            
            # Early stopping and checkpoint saving
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                patience_counter = 0
                self.save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

    def validate(self):
        val_loss, val_acc, val_dice, val_miou = self.validate_one_epoch()
        print(f"Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}, Val MIOU: {val_miou:.4f}")

    def test(self, checkpoint_path=None):
        # Load model checkpoint if specified
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        self.model.eval()
        test_loss, test_acc, test_dice, test_miou = self.test_one_epoch()
        print(f"Test Acc: {test_acc:.4f}, Test Dice: {test_dice:.4f}, Test MIOU: {test_miou:.4f}")
        return test_loss, test_acc, test_dice, test_miou
