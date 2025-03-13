import librosa
import numpy as np
import matplotlib.pyplot as plt
import pywt

def compute_and_visualize_scalogram(file_path):
    """
    Computes and visualizes the scalogram of an audio file using Continuous Wavelet Transform (CWT) with PyWavelets.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: CWT result matrix.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Generate a spectrogram
        D = np.abs(librosa.stft(y))

        # Use the mean frequency content across time
        spectrogram = np.mean(D, axis=1)

        # Define wavelet scales
        scales = np.arange(1, 128)

        # Apply Continuous Wavelet Transform (CWT) using PyWavelets
        cwt_result, _ = pywt.cwt(spectrogram, scales, 'morl')

        # Visualize the Scalogram
        plt.figure(figsize=(12, 8))
        plt.imshow(np.abs(cwt_result), extent=[0, len(y) / sr, 1, 128], cmap='viridis', aspect='auto')
        plt.colorbar(label="Magnitude")
        plt.title("Scalogram of the Audio (Using PyWavelets)")
        plt.xlabel("Time (s)")
        plt.ylabel("Scale")
        plt.show()

        return cwt_result

    except Exception as e:
        print(f"Error in Scalogram computation: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Compute and visualize the scalogram
    compute_and_visualize_scalogram(file_path)
