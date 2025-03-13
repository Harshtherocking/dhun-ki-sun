import librosa
import numpy as np

def compute_and_save_spectral_features(amp, sr, centroid_file="spectral_centroid_midi.csv", bandwidth_file="spectral_bandwidth_midi.csv"):
    """
    Computes spectral centroid, spectral bandwidth, and zero-crossing rate of an audio signal.
    Saves the centroid and bandwidth values as CSV files.

    Args:
        amp (numpy.ndarray): Audio waveform data.
        sr (int): Sample rate of the audio.
        centroid_file (str): Filename for saving spectral centroid values.
        bandwidth_file (str): Filename for saving spectral bandwidth values.

    Returns:
        dict: A dictionary containing spectral centroid, spectral bandwidth, and zero-crossing rate.
    """
    try:
        # Compute frame times
        frames = range(len(amp))
        t = librosa.frames_to_time(frames, sr=sr)

        # Compute Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=amp, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=amp, sr=sr)

        # Flatten arrays
        centroid = np.hstack(spectral_centroid)
        bandwidth = np.hstack(spectral_bandwidth)

        # Compute Zero Crossing Rate
        spectra = librosa.feature.zero_crossing_rate(y=amp)
        print(spectra)

        # Save Spectral Centroid and Bandwidth as CSV files
        np.savetxt(centroid_file, centroid.T, delimiter=",", fmt="%f")
        np.savetxt(bandwidth_file, bandwidth.T, delimiter=",", fmt="%f")

        print("Files saved successfully!")

        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "zero_crossing_rate": spectra
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"

    # Load audio
    amp, sr = librosa.load(file_path, sr=22050)

    # Compute and save spectral features
    compute_and_save_spectral_features(amp, sr)
