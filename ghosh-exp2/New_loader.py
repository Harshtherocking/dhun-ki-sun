import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np
from config import Config

class ThaatRagaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.segment_length = Config.SEGMENT_LENGTH
        self.sample_rate = Config.SAMPLE_RATE
        self.audio_files = []
        self.labels = []
        
        # Only process Asavari thaat
        thaat_path = os.path.join(root_dir, "Asavari")
        if os.path.exists(thaat_path):
            for raga in os.listdir(thaat_path):
                raga_path = os.path.join(thaat_path, raga)
                if os.path.isdir(raga_path):
                    for segment_folder in os.listdir(raga_path):
                        segment_path = os.path.join(raga_path, segment_folder)
                        if os.path.isdir(segment_path):
                            for file in os.listdir(segment_path):
                                if file.endswith('.mp3'):
                                    self.audio_files.append(os.path.join(segment_path, file))
                                    self.labels.append(0)  # Using 0 as label for Asavari
        
        print(f"Found {len(self.audio_files)} audio files in Asavari thaat")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Random segment selection for training
            if waveform.size(1) > self.segment_length:
                start_idx = torch.randint(0, waveform.size(1) - self.segment_length, (1,))
                waveform = waveform[:, start_idx:start_idx + self.segment_length]
            else:
                # Pad if too short
                pad_length = self.segment_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # Normalize audio
            waveform = waveform / (waveform.abs().max() + 1e-8)
            
            return waveform.squeeze(), torch.tensor(0)  # Using 0 as dummy label
            
        except Exception as e:
            print(f"Error loading file {audio_path}: {str(e)}")
            # Return a zero tensor in case of error
            return torch.zeros(self.segment_length), torch.tensor(0)
