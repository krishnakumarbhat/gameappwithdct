# import numpy as np
# import pywt
# from scipy.io import wavfile
# import matplotlib.pyplot as plt
# from collections import Counter
# import huffman

# # Read the audio file
# sampling_rate, data = wavfile.read('harvard.wav')

# # Perform wavelet transform
# coeffs = pywt.wavedec(data, 'db6')

# # Define the threshold for compression
# threshold = 1000

# # Apply thresholding to the wavelet coefficients
# compressed_coeffs = []
# for i, coeff in enumerate(coeffs):
#     compressed_coeff = pywt.threshold(coeff, threshold)
#     compressed_coeffs.append(compressed_coeff)

# # Flatten the coefficients
# flatten_coeffs = np.hstack(compressed_coeffs)

# # Convert the coefficients to tuples for counting
# tuple_coeffs = [tuple(coeff) for coeff in flatten_coeffs]

# # Perform Huffman encoding on the coefficients
# symbol_counts = dict(Counter(tuple_coeffs))
# huff_tree = huffman.codebook(symbol_counts)
# huffman_dict = dict(huff_tree)

# # Encode the coefficients using Huffman coding
# encoded_coeffs = [huffman_dict.get(symbol, '') for symbol in tuple_coeffs]
# binary_data = ''.join(encoded_coeffs)

# # Pad the binary data to have a multiple of 8 bits
# padding_length = 8 - len(binary_data) % 8
# binary_data += "0" * padding_length

# # Convert binary data to bytes
# byte_array = bytearray()
# for i in range(0, len(binary_data), 8):
#     byte = binary_data[i:i + 8]
#     byte_array.append(int(byte, 2))

# # Save the compressed audio file
# # wavfile.write('compressed.wav', sampling_rate, np.asarray(compressed_coeffs))
# # Save the compressed audio file
# flattened_coeffs = np.concatenate(compressed_coeffs)
# wavfile.write('compressed.wav', sampling_rate, flattened_coeffs)


# # Save the Huffman dictionary and compressed coefficients
# np.savez('compressed.npz', huffman_dict=huffman_dict, compressed_coeffs=compressed_coeffs)

import numpy as np
import pywt
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections import Counter
import huffman

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
flattened_coeffs = np.concatenate(compressed_coeffs)
wavfile.write('compressed.wav', sampling_rate, flattened_coeffs)

# Save the Huffman dictionary and compressed coefficients
np.savez('compressed.npz', huffman_dict=huffman_dict, compressed_coeffs=compressed_coeffs)

# Plot the original audio
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(data)
plt.title('Original Audio')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Plot the compressed audio
plt.subplot(2, 1, 2)
plt.plot(flattened_coeffs)
plt.title('Compressed Audio')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
