import torch
import os

class CheckpointManager:
    def __init__(self, checkpoint_dir="../checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save(self, model, optimizer, epoch, loss, filename="checkpoint.pth"):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load(self, model, optimizer, filename="checkpoint.pth"):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {checkpoint['epoch']}")
            return checkpoint['epoch'], checkpoint['loss']
        else:
            print("No checkpoint found, starting from scratch.")
            return 0, None
