import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Define dataset path
DATASET_PATH = "Thaat_and_Raga"
OUTPUT_PATH = "data/spectrograms"

# Function to convert audio to mel-spectrogram
def convert_to_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process dataset
for thaat in os.listdir(DATASET_PATH):
    thaat_path = os.path.join(DATASET_PATH, thaat)
    if not os.path.isdir(thaat_path):
        continue
    
    for raga in os.listdir(thaat_path):
        raga_path = os.path.join(thaat_path, raga)
        if not os.path.isdir(raga_path):
            continue
        
        for folder in os.listdir(raga_path):
            folder_path = os.path.join(raga_path, folder)
            if not os.path.isdir(folder_path):
                continue
            
            for file in os.listdir(folder_path):
                if file.endswith(".mp3") or file.endswith(".wav"):
                    audio_path = os.path.join(folder_path, file)
                    save_dir = os.path.join(OUTPUT_PATH, thaat, raga)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, file.replace(".mp3", ".png").replace(".wav", ".png"))
                    
                    convert_to_spectrogram(audio_path, save_path)
                    print(f"Processed: {audio_path} -> {save_path}")

