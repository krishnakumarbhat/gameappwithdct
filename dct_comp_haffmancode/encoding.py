import numpy as np
import pywt
from scipy.io import wavfile
import huffman
from collections import Counter
import matplotlib.pyplot as plt

# Read the audio file
sampling_rate, data = wavfile.read('harvard.wav')

# Perform wavelet transform
coeffs = pywt.wavedec(data, 'db6')

# Define the threshold for compression
threshold = 1000

# Apply thresholding to the wavelet coefficients
compressed_coeffs = []
for i, coeff in enumerate(coeffs):
    compressed_coeff = pywt.threshold(coeff, threshold)
    compressed_coeffs.append(compressed_coeff)

# Flatten the coefficients
flatten_coeffs = np.hstack(compressed_coeffs)

# Convert the coefficients to tuples for counting
tuple_coeffs = [tuple(coeff) for coeff in flatten_coeffs]

# Perform Huffman encoding on the coefficients
symbol_counts = dict(Counter(tuple_coeffs))
huff_tree = huffman.codebook(symbol_counts)
huffman_dict = dict(huff_tree)

# Encode the coefficients using Huffman coding
encoded_coeffs = [huffman_dict.get(symbol, '') for symbol in tuple_coeffs]
binary_data = ''.join(encoded_coeffs)

# Pad the binary data to have a multiple of 8 bits
padding_length = 8 - len(binary_data) % 8
binary_data += "0" * padding_length

# Convert binary data to bytes
byte_array = bytearray()
for i in range(0, len(binary_data), 8):
    byte = binary_data[i:i + 8]
    byte_array.append(int(byte, 2))

# Save the compressed audio file
np.savez('compressed.npz', huffman_dict=huffman_dict, compressed_coeffs=compressed_coeffs)

# Load the compressed data
data = np.load('compressed.npz', allow_pickle=True)
huffman_dict = data['huffman_dict'].item()
compressed_coeffs = data['compressed_coeffs']

# Convert the coefficients to tuples for decoding
tuple_coeffs = []
for coeff in compressed_coeffs:
    tuple_coeffs.append(tuple(coeff))

# Perform Huffman decoding
decoded_coeffs = ''
for code in tuple_coeffs:
    decoded_coeffs += huffman_dict.inverse.get(code, '')

# Convert binary data to bytes
byte_array = bytearray()
for i in range(0, len(decoded_coeffs), 8):
    byte = decoded_coeffs[i:i + 8]
    byte_array.append(int(byte, 2))

# Convert bytes to flattened coefficients
flatten_coeffs = np.frombuffer(byte_array, dtype=np.uint8)

# Reshape flattened coefficients
reshaped_coeffs = np.split(flatten_coeffs, len(compressed_coeffs))

# Perform wavelet reconstruction
reconstructed_data = pywt.waverec(reshaped_coeffs, 'db6')

# Save the decompressed audio file
wavfile.write('decompressed.wav', sampling_rate, reconstructed_data)

# Plotting
plt.figure(figsize=(10, 6))

# Original Audio Signal
plt.subplot(2, 1, 1)
plt.plot(data, color='b')
plt.title('Original Audio Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Decompressed Audio Signal
plt.subplot(2, 1, 2)
plt.plot(reconstructed_data, color='r')
plt.title('Decompressed Audio Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
