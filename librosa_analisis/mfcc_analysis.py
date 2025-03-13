import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def compute_and_visualize_mfcc(amp, sr, n_mfcc=13):
    """
    Computes and visualizes MFCCs from an audio signal.

    Args:
        amp (numpy.ndarray): Audio waveform data.
        sr (int): Sample rate of the audio.
        n_mfcc (int): Number of MFCC coefficients to extract (default: 13).

    Returns:
        numpy.ndarray: MFCCs matrix.
    """
    try:
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=amp, sr=sr, n_mfcc=n_mfcc)

        # Print MFCC shape
        print(f"MFCCs shape: {mfccs.shape}")

        # Visualize MFCCs
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='magma')
        plt.colorbar()
        plt.title('MFCCs')
        plt.show()

        return mfccs

    except Exception as e:
        print(f"Error in MFCC computation: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Load audio
    amp, sr = librosa.load(file_path, sr=22050)

    # Compute and visualize MFCCs
    compute_and_visualize_mfcc(amp, sr)
