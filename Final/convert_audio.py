import torch
import numpy as np
import librosa
import soundfile as sf
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pad_spectrogram(spec, target_size=(1025, 256)):
    """Pad spectrogram to match the expected size"""
    current_size = spec.shape
    pad_height = target_size[0] - current_size[0]
    
    if pad_height > 0:
        # Pad with zeros at the top
        padded_spec = np.pad(spec, ((0, pad_height), (0, 0)), mode='constant')
    else:
        # If spectrogram is larger, take the first target_size[0] frequencies
        padded_spec = spec[:target_size[0], :]
    
    return padded_spec

def griffinlim(spectrogram, n_iter=100, sr=22050, hop_length=256):
    """
    Griffin-Lim algorithm to convert spectrogram to audio
    """
    # Ensure proper shape
    if spectrogram.shape[0] != 1025:
        spectrogram = pad_spectrogram(spectrogram)
    
    # Scale the spectrogram to a reasonable range
    spectrogram = spectrogram * 100  # Adjust this scaling factor as needed
    
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
    S = np.abs(spectrogram)
    y = librosa.istft(S * angles, hop_length=hop_length, win_length=2048)

    for i in range(n_iter):
        D = librosa.stft(y, n_fft=2048, hop_length=hop_length, win_length=2048)
        angles = np.exp(1j * np.angle(D))
        y = librosa.istft(S * angles, hop_length=hop_length, win_length=2048)

    return y

def convert_spectrogram_to_audio(spec_tensor, sr=22050, hop_length=256):
    """
    Convert a spectrogram tensor to audio using Griffin-Lim algorithm
    """
    # Convert to numpy and denormalize
    spec = spec_tensor.squeeze().cpu().numpy()
    
    # Scale to proper range
    spec = (spec - spec.min()) / (spec.max() - spec.min())
    
    # Apply Griffin-Lim algorithm
    audio = griffinlim(spec, n_iter=100, sr=sr, hop_length=hop_length)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    return audio

def process_all_samples(input_dir='generated_samples', output_dir='generated_audio', sr=22050):
    """
    Process all spectrogram samples in the input directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .pt files
    spec_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
    
    logger.info(f"Found {len(spec_files)} spectrogram files to process")
    
    for spec_file in tqdm(spec_files, desc="Converting to audio"):
        try:
            # Load spectrogram tensor
            spec_path = os.path.join(input_dir, spec_file)
            spec_tensor = torch.load(spec_path, weights_only=True)
            
            # Convert to audio
            audio = convert_spectrogram_to_audio(spec_tensor)
            
            # Save audio file
            output_path = os.path.join(output_dir, f"{spec_file[:-3]}.wav")
            sf.write(output_path, audio, sr)
            
        except Exception as e:
            logger.error(f"Error processing {spec_file}: {str(e)}")
            logger.error(f"Spectrogram shape: {spec_tensor.shape}")

def main():
    logger.info("Starting audio conversion process...")
    process_all_samples()
    logger.info("Done! Check the 'generated_audio' directory for the results.")

if __name__ == '__main__':
    main()
