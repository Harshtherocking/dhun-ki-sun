from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn.functional as F
import gc
import logging

logger = logging.getLogger(__name__)

class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, target_size=(128, 256)):
        self.spectrogram_dir = spectrogram_dir
        self.files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.pt')]
        self.target_size = target_size
        logger.info(f"Found {len(self.files)} spectrogram files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file_path = os.path.join(self.spectrogram_dir, self.files[idx])
            # Use map_location='cpu' to ensure tensors are loaded in CPU memory
            spectrogram = torch.load(file_path, map_location='cpu', weights_only=True)
            
            # Add channel dimension if not present
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)
            
            # Resize to target size
            if spectrogram.size(1) != self.target_size[0] or spectrogram.size(2) != self.target_size[1]:
                spectrogram = F.interpolate(
                    spectrogram.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Ensure the tensor is contiguous and in CPU memory
            spectrogram = spectrogram.contiguous()
            
            # Clear some memory
            if idx % 100 == 0:
                gc.collect()
                
            return spectrogram
        except Exception as e:
            logger.error(f"Error loading file {self.files[idx]}: {str(e)}")
            raise

def get_dataloader(batch_size=4, num_workers=8, collate_fn=None):
    spectrogram_path = "/home/amithk/Desktop/Dhun_ki_sun/FinalModel/Diffusion/Spectrogram_data"
    dataset = SpectrogramDataset(spectrogram_path)
    return DataLoader(
        dataset,
        batch_size=4,  # Fixed batch size of 4
        shuffle=True,
        num_workers=8,  # 8 workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
