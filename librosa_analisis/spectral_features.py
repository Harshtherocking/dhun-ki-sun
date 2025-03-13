import librosa
import numpy as np

def extract_spectral_features(amp, sr, output_file="output.csv"):

    try:
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=amp, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=amp, sr=sr)

        # Flatten and stack spectral features
        stacked_array = np.hstack(spectral_centroid)

        # Save the stacked array as a CSV file
        np.savetxt(output_file, stacked_array.T, delimiter=",", fmt="%f")
        print(f"File saved successfully as {output_file}!")

        # Print extracted features
        print("Spectral Centroid:", spectral_centroid)
        print("Spectral Bandwidth:", spectral_bandwidth)

        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage (only runs if this script is executed directly)
if __name__ == "__main__":
    import librosa.display

    file_path =r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"
    amp, sr = librosa.load(file_path, sr=22050)

    extract_spectral_features(amp, sr)
