# /* Final code*/

import os
import torch  # type: ignore
import torchaudio  # type: ignore
import torch.nn.functional as F  # type: ignore
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader  # type: ignore

class TRFDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_length=220500):  # 5 sec at 44.1kHz
        self.root_dir = root_dir
        self.transform = transform
        self.target_length = target_length  # Fixed length for padding/truncation
        self.data = []
        self.raga_to_idx = {}

        # Traverse folders: Thaat -> Raga -> Song Segments
        for thaat_name in sorted(os.listdir(root_dir)):
            thaat_path = os.path.join(root_dir, thaat_name)
            if not os.path.isdir(thaat_path):
                continue

            for raga_idx, raga_name in enumerate(sorted(os.listdir(thaat_path))):
                raga_path = os.path.join(thaat_path, raga_name)
                if not os.path.isdir(raga_path):
                    continue

                # Map Raga to an integer
                self.raga_to_idx[raga_name] = raga_idx  

                # Traverse Song folders and collect audio paths
                for song_folder in os.listdir(raga_path):
                    song_path = os.path.join(raga_path, song_folder)
                    if os.path.isdir(song_path):
                        for audio_file in os.listdir(song_path):
                            if audio_file.endswith(('.wav', '.mp3')):  # Check for audio files
                                audio_path = os.path.join(song_path, audio_file)
                                self.data.append((audio_path, raga_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, raga_label = self.data[idx]
        
        # Load audio file
        # if audio_path.endswith('.mp3', '.wav'):
        #     waveform, sample_rate = torchaudio.load(audio_path, normalize=True )
        waveform, sample_rate = torchaudio.load(audio_path, format="mp3", normalize=True)

        # Handle variable-length audio: Pad or Truncate
        num_channels, num_samples = waveform.shape
        if num_samples < self.target_length:
            pad = self.target_length - num_samples
            waveform = F.pad(waveform, (0, pad))  # Pad at the end
        else:
            waveform = waveform[:, :self.target_length]  # Truncate

        # Apply any transformation if provided (e.g., resampling, normalization)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, raga_label

def plot_waveform(waveform, sample_idx=0):
    """Plot the waveform of a given sample."""
    plt.figure(figsize=(12, 4))
    plt.plot(waveform[sample_idx].t().numpy())
    plt.title(f"Waveform of Sample {sample_idx}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

if __name__ == "__main__":
    # Define dataset path
    root_dir = r"C:\zzz\Solution_chall\Thaat and Raga Forest (TRF) Dataset Output"
    root_dir =  r"C:\Users\harsh\OneDrive\Desktop\dhun-ki-sun\Thaat\poorvi\adana\0"

    # Create dataset instance
    trf_dataset = TRFDataset(root_dir, target_length=220500)  # Target: 5 sec @ 44.1kHz

    # Create DataLoader
    trf_loader = DataLoader(trf_dataset, batch_size=2, shuffle=True)

    # Inspect a batch and plot a waveform
    for x, raga_y in trf_loader:
        print("Waveform (x):", x.shape)  # Shape should be (batch_size, channels, samples)
        print("Raga Labels (y2):", raga_y)  # Integer labels for Raga

        # Plot the waveform of the first sample in the batch
        plot_waveform(x[0])
        plot_waveform(x[1])

        # X is the waveform and y is the raga label
        # X is the list of waveforms according to the batch size
        break
