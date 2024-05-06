import numpy as np
import pywt
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections import Counter
import huffman
from sklearn.tree import DecisionTreeRegressor

# Read the audio file
sampling_rate, data = wavfile.read('harvard.wav')

# Perform wavelet transform 
coeffs = pywt.wavedec(data, 'db6')

# Use a machine learning model (Decision Tree Regressor) to predict the threshold
# Train the model on the wavelet coefficients to predict optimal thresholds
regressors = []
thresholds = []

for i, coeff in enumerate(coeffs):
    regressor = DecisionTreeRegressor()
    flatten_coeff = coeff.flatten()
    x_train = np.arange(len(flatten_coeff)).reshape(-1, 1)
    y_train = flatten_coeff
    regressor.fit(x_train, y_train)
    regressors.append(regressor)
    thresholds.append(np.mean(np.abs(regressor.predict(x_train) - y_train)))

# Apply adaptive thresholding to the wavelet coefficients
compressed_coeffs = []
for i, coeff in enumerate(coeffs):
    compressed_coeff = pywt.threshold(coeff, thresholds[i], mode='soft')
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
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(data)
plt.title('Original Audio')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Plot the wavelet coefficients
plt.subplot(4, 1, 2)
for i, coeff in enumerate(coeffs):
    plt.plot(coeff, label=f'Level {i}')
plt.legend()
plt.title('Wavelet Coefficients')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Plot the compressed coefficients
plt.subplot(4, 1, 3)
for i, coeff in enumerate(compressed_coeffs):
    plt.plot(coeff, label=f'Level {i}')
plt.legend()
plt.title('Compressed Coefficients')
plt.xlabel('Sample')
plt.ylabel('Amplitude') 

# Plot the Huffman encoding statistics
plt.subplot(4, 1, 4)
symbol_frequencies = list(symbol_counts.values())
plt.bar(range(len(symbol_frequencies)), symbol_frequencies)
plt.title('Huffman Encoding Statistics')
plt.xlabel('Symbol')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Print the results
# print("Original Audio Data:")
# print(data)

print("\nCompressed Audio Data:")
print(flattened_coeffs)

print("\nHuffman Dictionary:")
#print(huffman_dict)

print("\nHuffman Encoding Statistics:")
#print(symbol_counts)

print("\nNumber of Bits before Compression:", len(data) * 16)  # Assuming 16 bits per sample in the original audio
print("Number of Bits after Compression:", len(byte_array) * 8)
compression_ratio = (len(byte_array) * 8) / (len(data) * 16)
print("Compression Ratio:", compression_ratio)
