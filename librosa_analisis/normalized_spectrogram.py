import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def compute_and_visualize_normalized_spectrogram(file_path):
    """
    Computes and visualizes the normalized spectrogram of an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Normalized spectrogram matrix.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Compute Spectrogram
        D = np.abs(librosa.stft(y))

        # Normalize Spectrogram (0 to 1)
        D_norm = (D - np.min(D)) / (np.max(D) - np.min(D))

        # Display Normalized Spectrogram
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(D_norm, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(label='Normalized Magnitude')
        plt.title('Normalized Spectrogram')
        plt.show()

        print("Spectrogram shape:", D_norm.shape)
        return D_norm

    except Exception as e:
        print(f"Error in spectrogram normalization: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Compute and visualize normalized spectrogram
    compute_and_visualize_normalized_spectrogram(file_path)
