import wave
import pygame

def play_wav(filename):
    # Open the WAV file
    wave_file = wave.open(filename, 'rb')

    # Initialize Pygame mixer
    pygame.mixer.init()

    # Load the WAV file
    pygame.mixer.music.load(wave_file)

    # Play the WAV file
    pygame.mixer.music.play()

    # Wait until the sound finishes playing
    while pygame.mixer.music.get_busy():
        continue

    # Close the WAV file
    wave_file.close()

# Replace 'your_wav_file.wav' with the path to your WAV file
play_wav('harvard.wav')
