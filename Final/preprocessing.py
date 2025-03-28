import torchaudio
import torchaudio.transforms as transforms
import torch
import os
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import hashlib

# Set up logging
log_dir = "preprocessing_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"preprocessing_{timestamp}.log")
error_file = os.path.join(log_dir, f"error_summary_{timestamp}.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Paths and constants
SOURCE_DIR = "/home/amithk/Desktop/Dhun_ki_sun/Thaat_and_Raga/Thaat_and_Raga/Asavari"
DESTINATION_DIR = "/home/amithk/Desktop/Dhun_ki_sun/FinalModel/Diffusion/Spectrogram_data"
SAMPLE_RATE = 16000
TARGET_LENGTH = 5 * SAMPLE_RATE  # 5 seconds of audio
NUM_WORKERS = 8

# Error tracking
error_summary = {
    "failed_files": [],
    "success_count": 0,
    "error_types": {},
    "skipped_files": []
}

def get_unique_filename(file_path):
    """Generate unique filename preserving directory structure."""
    rel_path = os.path.relpath(file_path, SOURCE_DIR)
    # Replace directory separators with underscores
    unique_name = rel_path.replace(os.sep, '_')
    # Remove .mp3 extension
    unique_name = os.path.splitext(unique_name)[0]
    return unique_name + ".pt"

def pad_or_trim(waveform, target_length):
    """Pad or trim the audio waveform to target length."""
    current_length = waveform.size(1)
    
    if current_length > target_length:
        # Take the middle section
        start = (current_length - target_length) // 2
        return waveform[:, start:start + target_length]
    elif current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        pad_left = padding // 2
        pad_right = padding - pad_left
        return torch.nn.functional.pad(waveform, (pad_left, pad_right))
    
    return waveform

def validate_audio(waveform, sample_rate, min_length=0.1):
    """Validate audio data."""
    if waveform is None or sample_rate is None:
        return False, "Invalid audio data"
    
    duration = waveform.size(1) / sample_rate
    if duration < min_length:
        return False, f"Audio too short: {duration:.2f}s"
    
    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        return False, "Audio contains NaN or Inf values"
    
    if waveform.abs().max() < 1e-6:
        return False, "Audio is silent"
    
    return True, None

def process_audio_file(file_path):
    """Process a single audio file."""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Validate audio
        is_valid, error_msg = validate_audio(waveform, sample_rate)
        if not is_valid:
            error_summary["skipped_files"].append({
                "file": file_path,
                "reason": error_msg
            })
            return False
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            resampler = transforms.Resample(sample_rate, SAMPLE_RATE).to(device)
            waveform = resampler(waveform.to(device))
        else:
            waveform = waveform.to(device)
        
        # Pad or trim to target length
        waveform = pad_or_trim(waveform, TARGET_LENGTH)
        
        # Create and apply mel spectrogram transform
        mel_transform = transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            f_min=20,
            f_max=8000
        ).to(device)
        
        mel_spectrogram = mel_transform(waveform)
        mel_spectrogram = transforms.AmplitudeToDB()(mel_spectrogram)
        
        # Normalize
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / (mel_spectrogram.std() + 1e-8)
        
        # Generate unique filename
        filename = get_unique_filename(file_path)
        save_path = os.path.join(DESTINATION_DIR, filename)
        
        # Save spectrogram
        torch.save(mel_spectrogram.cpu(), save_path)
        
        # Verify file was saved
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            error_summary["success_count"] += 1
            return True
        else:
            raise Exception("File not saved properly")

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        if error_type not in error_summary["error_types"]:
            error_summary["error_types"][error_type] = 0
        error_summary["error_types"][error_type] += 1
        
        error_summary["failed_files"].append({
            "file": file_path,
            "error_type": error_type,
            "error_message": error_msg,
            "traceback": traceback.format_exc()
        })
        
        logging.error(f"Error processing {file_path}: {error_msg}")
        return False

def find_audio_files(directory):
    """Find all MP3 files recursively."""
    audio_files = []
    logging.info(f"Searching for audio files in: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp3"):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    logging.info(f"Found {len(audio_files)} audio files")
    return audio_files

def save_error_summary():
    """Save error summary to JSON file."""
    with open(error_file, 'w') as f:
        json.dump(error_summary, f, indent=4)
    logging.info(f"Error summary saved to {error_file}")

def clear_destination():
    """Clear the destination directory of existing .pt files."""
    if os.path.exists(DESTINATION_DIR):
        for file in os.listdir(DESTINATION_DIR):
            if file.endswith(".pt"):
                os.remove(os.path.join(DESTINATION_DIR, file))
        logging.info("Cleared existing spectrogram files")

def main():
    start_time = datetime.now()
    logging.info("Starting preprocessing pipeline")
    torch.cuda.empty_cache()

    # Clear destination directory
    clear_destination()

    # Ensure destination directory exists
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    logging.info(f"Destination directory: {DESTINATION_DIR}")

    # Find all audio files
    audio_files = find_audio_files(SOURCE_DIR)
    total_files = len(audio_files)
    
    # Process files with progress bar
    with tqdm(total=total_files, desc="Processing files") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Process files in parallel
            futures = []
            for file_path in audio_files:
                future = executor.submit(process_audio_file, file_path)
                futures.append(future)
            
            # Update progress as files complete
            for future in futures:
                success = future.result()
                pbar.update(1)
                
                # Clear GPU cache periodically
                if pbar.n % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Verify final count
    actual_saved = len([f for f in os.listdir(DESTINATION_DIR) if f.endswith('.pt')])
    if actual_saved != error_summary["success_count"]:
        logging.error(f"Mismatch in saved files! Counted: {error_summary['success_count']}, Actually saved: {actual_saved}")
        error_summary["success_count"] = actual_saved

    # Log summary statistics
    end_time = datetime.now()
    duration = end_time - start_time
    success_rate = (error_summary["success_count"] / total_files) * 100 if total_files > 0 else 0
    
    logging.info("\nProcessing Summary:")
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Successfully processed: {error_summary['success_count']}")
    logging.info(f"Failed to process: {len(error_summary['failed_files'])}")
    logging.info(f"Skipped files: {len(error_summary['skipped_files'])}")
    logging.info(f"Success rate: {success_rate:.2f}%")
    logging.info(f"Total processing time: {duration}")
    
    if error_summary["error_types"]:
        logging.info("\nError type distribution:")
        for error_type, count in error_summary["error_types"].items():
            logging.info(f"{error_type}: {count} occurrences")
    
    save_error_summary()
    logging.info(f"Detailed logs available in: {log_file}")
    logging.info(f"Error summary available in: {error_file}")

if __name__ == "__main__":
    main()
