import librosa
import os

def load_audio_file(file_path, sample_rate=22050):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the audio file
        amp, sr = librosa.load(file_path, sr=sample_rate)
        print(f"Audio Shape: {amp.shape}, Sample Rate: {sr}")
        return amp, sr

    except FileNotFoundError as e:
        print(e)
    except librosa.util.exceptions.ParameterError as param_error:
        print(f"Parameter error: {param_error}")
    except Exception as e:
        print(f"Error loading the audio file: {e}")

    return None, None  # Return None if loading fails

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\amits\IdeaProjects\music manipulation\spilleted\Avsari\Adana\0\0_20250302_174503_part001.mp3"  # Replace with actual file path
    load_audio_file(file_path)
