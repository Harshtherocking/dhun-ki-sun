import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def compute_and_visualize_normalized_mfcc(file_path, n_mfcc=13):
    """
    Computes and visualizes normalized MFCCs from an audio file.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC coefficients to extract (default: 13).

    Returns:
        numpy.ndarray: Normalized MFCC matrix.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Normalize MFCC (0 to 1)
        mfccs_norm = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))

        # Display Normalized MFCC
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mfccs_norm, sr=sr, x_axis='time', cmap='magma')
        plt.colorbar(label='Normalized Coefficients')
        plt.title('Normalized MFCC')
        plt.show()

        print("MFCC shape:", mfccs_norm.shape)
        return mfccs_norm

    except Exception as e:
        print(f"Error in MFCC normalization: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Compute and visualize normalized MFCC
    compute_and_visualize_normalized_mfcc(file_path)