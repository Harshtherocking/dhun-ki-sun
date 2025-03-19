import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class TRFDataset(Dataset):
    def __init__(self, root_dir, target_length=220500):
        self.root_dir = root_dir
        self.target_length = target_length
        self.data = []
        self.song_to_idx = {}
        song_id = 0

        for thaat_name in sorted(os.listdir(root_dir)):
            thaat_path = os.path.join(root_dir, thaat_name)
            if not os.path.isdir(thaat_path):
                continue

            for raga_name in sorted(os.listdir(thaat_path)):
                raga_path = os.path.join(thaat_path, raga_name)
                if not os.path.isdir(raga_path):
                    continue

                for song_folder in sorted(os.listdir(raga_path)):
                    song_path = os.path.join(raga_path, song_folder)
                    if os.path.isdir(song_path):
                        self.song_to_idx[song_folder] = song_id  

                        for idx, audio_file in enumerate(sorted(os.listdir(song_path))):
                            if audio_file.endswith(('.wav', '.mp3')):
                                audio_path = os.path.join(song_path, audio_file)
                                self.data.append((song_id, idx, audio_path))  
                        song_id += 1  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        song_id, clip_idx, audio_path = self.data[idx]

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)  # Standardize

        # Ensure Mono (Convert Stereo to Single Channel)
        if waveform.shape[0] > 1:  
            waveform = torch.mean(waveform, dim=0, keepdim=True)  

        # Ensure Fixed Length (Pad or Truncate)
        num_samples = waveform.shape[1]
        if num_samples < self.target_length:
            pad = self.target_length - num_samples
            waveform = F.pad(waveform, (0, pad))  
        else:
            waveform = waveform[:, :self.target_length]  

        return song_id, clip_idx, waveform.squeeze(0)  
