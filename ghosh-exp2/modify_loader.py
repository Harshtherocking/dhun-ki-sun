import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np

class ModifiedThaatRagaDataset(Dataset):
    def __init__(self, root_dir, segment_length=55125):  # Half the original length
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.audio_files = []
        self.labels = []
        
        # Only process first thaat (Asavari)
        thaat_path = os.path.join(root_dir, "Asavari")
        if os.path.exists(thaat_path):
            for raga in os.listdir(thaat_path):
                raga_path = os.path.join(thaat_path, raga)
                if os.path.isdir(raga_path):
                    for file in os.listdir(raga_path):
                        if file.endswith(('.mp3', '.wav')):
                            self.audio_files.append(os.path.join(raga_path, file))
                            self.labels.append(0)  # Single thaat, so all labels are 0
            break

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 44100:
            waveform = torchaudio.transforms.Resample(sample_rate, 44100)(waveform)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Randomly select a segment if waveform is long enough
        if waveform.size(1) > self.segment_length:
            start_idx = torch.randint(0, waveform.size(1) - self.segment_length, (1,))
            waveform = waveform[:, start_idx:start_idx + self.segment_length]
        else:
            # Pad if too short
            pad_length = self.segment_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return torch.tensor(self.labels[idx]), waveform

# Save the modified dataset class
with open('Modified_loader.py', 'w') as f:
    f.write("""
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np

""")
    f.write("\n".join([line for line in ModifiedThaatRagaDataset.__doc__.split('\n') if line.strip()]))
    f.write("\n\n")
    import inspect
    f.write(inspect.getsource(ModifiedThaatRagaDataset))

# Update VAEtrain.py to use the modified dataset
with open('VAEtrain.py', 'r') as f:
    content = f.read()

content = content.replace('from New_loader import ThaatRagaDataset', 'from Modified_loader import ModifiedThaatRagaDataset')
content = content.replace('dataset = ThaatRagaDataset', 'dataset = ModifiedThaatRagaDataset')

with open('VAEtrain.py', 'w') as f:
    f.write(content)

print("Modified dataset and training script have been updated.")
