import librosa  # type: ignore
import librosa.display  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to add Gaussian noise to the waveform
def add_gaussian_noise(audio, noise_level=0.02):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

# Function to normalize the waveform to [-1, 1]
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

# Function to show the difference between the original and the noisy audio
def show_waveform(amp, amp_noisy, sr, amp_noisy_normalized):
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(amp, sr=sr)
    plt.title('Original Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(amp_noisy, sr=sr)
    plt.title('Noisy Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.figure(figsize=(15, 5))
    plt.plot(amp_noisy, label='Noisy', alpha=0.7)
    plt.plot(amp, label='Original', alpha=0.7)
    plt.title('Difference between Original and Noisy Waveforms')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    # plot the normalized noisy audio
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(amp_noisy_normalized, sr=sr)
    plt.title('Noisy Waveform (After Normalization)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    

    plt.show()

def main():
    file_path = '0_084.mp3'  # Replace with the actual file name

    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the audio file
        amp, sr = librosa.load(file_path, sr=22050)
        print(f"Audio Shape: {amp.shape}, Sample Rate: {sr}")

        # Apply Gaussian noise
        amp_noisy = add_gaussian_noise(amp, noise_level=0.02)

        # Normalize the noisy audio
        amp_noisy_normalized = normalize_audio(amp_noisy)

        # Plot waveforms
        show_waveform(amp, amp_noisy, sr, amp_noisy_normalized)


    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except librosa.ParameterError as param_error:
        print(f"Parameter error: {param_error}")
    except Exception as e:
        print(f"Error loading the audio file: {e}")

if __name__ == "__main__":
    main()
