import torchaudio
import torch
import os

def spectrogram_to_audio(input_dir, output_dir, sample_rate=16000):
    """Converts saved spectrograms back into audio waveforms."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    inverse_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=201, n_mels=128)
    griffin_lim = torchaudio.transforms.GriffinLim(n_iter=32)
    
    for file in os.listdir(input_dir):
        if file.endswith('.pt'):
            spectrogram = torch.load(os.path.join(input_dir, file))
            spectrogram = inverse_mel_transform(spectrogram)
            waveform = griffin_lim(spectrogram)
            
            output_filepath = os.path.join(output_dir, file.replace('.pt', '.wav'))
            torchaudio.save(output_filepath, waveform, sample_rate)
            print(f"Converted {file} -> {output_filepath}")

# Example Usage
# spectrogram_to_audio("../data/spectrograms", "../output/audio")