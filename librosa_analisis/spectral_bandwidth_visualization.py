import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_waveform_with_spectral_bandwidth(amp, sr, spectral_bandwidth):
    """
    Plots the waveform and spectral bandwidth of an audio signal.

    Args:
        amp (numpy.ndarray): Audio waveform data.
        sr (int): Sample rate of the audio.
        spectral_bandwidth (numpy.ndarray): Spectral bandwidth values.
    """
    try:
        # Get number of frames and convert to time
        frames = range(spectral_bandwidth.shape[1])
        t = librosa.frames_to_time(frames, sr=sr)

        # Plot waveform with spectral bandwidth
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(amp, sr=sr, alpha=0.6, label="Waveform")
        plt.plot(t, spectral_bandwidth.T / np.max(spectral_bandwidth), marker='o', color='b', label='Spectral Bandwidth')

        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.title("Waveform with Spectral Features")
        plt.show()

    except Exception as e:
        print(f"Error in visualization: {e}")

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Load audio
    amp, sr = librosa.load(file_path, sr=22050)

    # Compute Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=amp, sr=sr)

    # Plot waveform with spectral bandwidth
    plot_waveform_with_spectral_bandwidth(amp, sr, spectral_bandwidth)
