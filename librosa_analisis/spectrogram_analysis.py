import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def compute_and_visualize_spectrogram(amp, sr):
    """
    Computes and visualizes the Short-Time Fourier Transform (STFT) spectrogram
    and chroma feature of an audio signal.

    Args:
        amp (numpy.ndarray): Audio waveform data.
        sr (int): Sample rate of the audio.

    Returns:
        tuple: (Spectrogram matrix, Chroma feature matrix)
    """
    try:
        # Compute Short-Time Fourier Transform (STFT)
        D = librosa.stft(amp)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

        # Plot Spectrogram
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='plasma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()

        # Compute Chroma feature
        chroma = librosa.feature.chroma_stft(y=amp, sr=sr)

        # Plot Chroma feature
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(chroma, x_axis="time", cmap="magma")
        plt.colorbar()
        plt.title("Chroma Feature")
        plt.show()

        print("Spectrogram Shape (Frequency bins, Time frames):", D.shape)
        return D, chroma

    except Exception as e:
        print(f"Error in spectrogram analysis: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Load audio
    amp, sr = librosa.load(file_path, sr=22050)

    # Compute and visualize spectrogram
    compute_and_visualize_spectrogram(amp, sr)
