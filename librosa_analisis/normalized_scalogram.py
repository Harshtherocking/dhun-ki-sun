import librosa
import numpy as np
import matplotlib.pyplot as plt
import pywt

def compute_and_visualize_normalized_scalogram(file_path):
    """
    Computes and visualizes the normalized scalogram of an audio file using Continuous Wavelet Transform (CWT).

    Args:
        file_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Normalized scalogram matrix.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Define wavelet scales and the wavelet type (using 'cmor' for continuous Morlet wavelet)
        widths = np.arange(1, 128)  # Number of scales
        wavelet = 'cmor1.5-1.0'     # Complex Morlet wavelet

        # Compute the scalogram using PyWavelets
        coeffs, _ = pywt.cwt(y, widths, wavelet, sampling_period=1/sr)

        # Get the magnitude of the coefficients
        scalogram = np.abs(coeffs)

        # Normalize the scalogram (0 to 1)
        scalogram_norm = (scalogram - np.min(scalogram)) / (np.max(scalogram) - np.min(scalogram))

        # Display the normalized scalogram
        plt.figure(figsize=(12, 8))
        plt.imshow(scalogram_norm, extent=[0, len(y) / sr, 1, 128], cmap='plasma', aspect='auto')
        plt.colorbar(label="Normalized Magnitude")
        plt.title("Normalized Scalogram (Mono Audio)")
        plt.xlabel("Time (s)")
        plt.ylabel("Scale")
        plt.show()

        # Print the shape of the scalogram
        print("Scalogram shape (Scales, Time frames):", scalogram_norm.shape)

        return scalogram_norm

    except Exception as e:
        print(f"Error in Scalogram computation: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Compute and visualize the normalized scalogram
    compute_and_visualize_normalized_scalogram(file_path)
