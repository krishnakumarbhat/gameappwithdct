import wave
import numpy as np
import matplotlib.pyplot as plt

def plot_wav(filename):
    # Open the WAV file
    wave_file = wave.open(filename, 'rb')

    # Get the number of frames
    num_frames = wave_file.getnframes()

    # Read the frames as bytes
    raw_data = wave_file.readframes(num_frames)

    # Convert the raw data to numpy array
    data = np.frombuffer(raw_data, dtype=np.int16)

    # Get the frame rate (sample rate)
    frame_rate = wave_file.getframerate()

    # Create a time axis for the waveform
    time = np.linspace(0, len(data) / frame_rate, num=len(data))

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True)
    plt.show()

    # Close the WAV file
    wave_file.close()

# Replace 'your_wav_file.wav' with the path to your WAV file
plot_wav('compressed.wav')
