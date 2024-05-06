import numpy as np
import pywt
from scipy.io import wavfile
import huffman

# Load the compressed data
data = np.load('compressed.npz', allow_pickle=True)
huffman_dict = data['huffman_dict'].item()
compressed_coeffs = data['compressed_coeffs']

# Convert floating-point coefficients to bytes
compressed_coeffs = compressed_coeffs.astype(np.uint8)

# Convert bytes to binary data
binary_data = ''.join(format(byte, '08b') for byte in compressed_coeffs.flatten())

# Remove padding from binary data
padding_length = int(binary_data[-8:], 2)
binary_data = binary_data[:-padding_length]

# Perform Huffman decoding
decoded_coeffs = ''
current_code = ''
for bit in binary_data:
    current_code += bit
    if current_code in huffman_dict:
        decoded_coeffs += huffman_dict[current_code]
        current_code = ''

# Convert coefficients to tuples
tuple_coeffs = []
current_tuple = ()
for code in decoded_coeffs:
    current_tuple += int(code),
    if len(current_tuple) == 3:
        tuple_coeffs.append(current_tuple)
        current_tuple = ()

# Perform wavelet reconstruction
reconstructed_data = pywt.waverec(tuple_coeffs, 'db6')

# Save the decompressed audio file
sampling_rate = 44100  # Replace with the correct sampling rate of your audio
wavfile.write('decompressed.wav', sampling_rate, reconstructed_data)
