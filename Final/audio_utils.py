import torchaudio
import torch
import torchaudio.functional as F
import torchaudio.transforms as T

def resample_audio(waveform, orig_sr, new_sr):
    """Resamples an audio waveform to a new sample rate."""
    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)(waveform)

def reduce_noise(waveform, noise_level=0.02):
    """Applies simple noise reduction by thresholding low amplitudes."""
    return torch.where(torch.abs(waveform) > noise_level, waveform, torch.tensor(0.0))

def normalize_audio(waveform):
    """Normalizes waveform amplitude to the range [-1, 1]."""
    return waveform / torch.max(torch.abs(waveform))

def augment_pitch(waveform, n_steps):
    """Shifts pitch by n_steps (positive = higher pitch, negative = lower pitch)."""
    return T.PitchShift(sample_rate=16000, n_steps=n_steps)(waveform)

# Example Usage
# waveform, sr = torchaudio.load("example.wav")
# waveform = resample_audio(waveform, sr, 16000)
# waveform = reduce_noise(waveform)
# waveform = normalize_audio(waveform)