import pywt
import numpy as np
import soundfile as sf

# Load the speech signal
speech_signal, sample_rate = sf.read('speech.mp3')

# Perform wavelet decomposition
level = 4  # Number of decomposition levels
wavelet = 'db4'  # Wavelet type
coeffs = pywt.wavedec(speech_signal, wavelet, level=level)

# Set the compression ratio (percentage of coefficients to keep) 
compression_ratio = 0.5

# Calculate the number of coefficients to keep
total_coeffs = sum(len(c) for c in coeffs)
num_coeffs_to_keep = int(total_coeffs * compression_ratio)

# Sort and truncate coefficients
flattened_coeffs = np.concatenate(coeffs)
sorted_coeffs = np.sort(np.abs(flattened_coeffs))
threshold = sorted_coeffs[-num_coeffs_to_keep]
flattened_coeffs[np.abs(flattened_coeffs) < threshold] = 0

# Perform wavelet reconstruction
reconstructed_signal = pywt.waverec(flattened_coeffs, wavelet)

# Save the compressed speech signal
sf.write('compressed_speech.mp3', reconstructed_signal, sample_rate)
