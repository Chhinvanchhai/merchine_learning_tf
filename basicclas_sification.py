# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt


def load_labels(file_path):
    with open(file_path, 'rb') as file:
        # Read the magic number and number of labels
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        num_labels = int.from_bytes(file.read(4), byteorder='big')
        
        # Read the labels
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    
    return labels

file_path = './train-labels-idx1-ubyte'  # Adjust the path to your extracted file
labels = load_labels(file_path)

def load_images(file_path):
    with open(file_path, 'rb') as file:
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        num_images = int.from_bytes(file.read(4), byteorder='big')
        num_rows = int.from_bytes(file.read(4), byteorder='big')
        num_cols = int.from_bytes(file.read(4), byteorder='big')
        
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

image_file_path = './train-images-idx3-ubyte'
images = load_images(image_file_path)
print(f'Loaded {len(images)} images')


print(f'Loaded {len(labels)} labels')
index = 1
plt.imshow(images[index], cmap='gray')
plt.title(f'Label: {labels[index]}')
plt.show()


# Build a simple neural network model
# model = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(images, labels, epochs=5)

# # Evaluate the model
# loss, accuracy = model.evaluate(images, labels)
# print(f'Test accuracy: {accuracy}')
