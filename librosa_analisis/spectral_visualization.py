import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def visualize_spectral_features(amp, sr):
    try:
        # Compute Zero Crossing Rate
        spectra = librosa.feature.zero_crossing_rate(y=amp)

        # Spectral visualization
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectra, x_axis="time")
        plt.colorbar()
        plt.title("Zero Crossing Rate Spectrogram")
        plt.show()

        # Waveform visualization
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(amp, sr=sr)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

        return spectra

    except Exception as e:
        print(f"Error in visualization: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Load audio
    amp, sr = librosa.load(file_path, sr=22050)

    # Visualize features
    visualize_spectral_features(amp, sr)
