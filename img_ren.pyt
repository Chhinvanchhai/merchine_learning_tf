import numpy as np
import matplotlib.pyplot as plt

# Create a simple 2x2 image with 3 channels (RGB)
image = np.array(
    [ 
        [
            [255, 0, 0], 
            [0, 255, 0]
        ],   # Red, Green
        [
            [0, 0, 255], 
            [255, 255, 0]
        ],
        [
            [255, 255, 255], 
            [255, 0, 255]
        ]
    ], dtype=np.uint8)  # Blue, Yellow
# shape(3,2,2)

print("Image Array:")
print(image)
print("Shape of the image:", image.shape)

# Display the imageclear
plt.imshow(image)
plt.title("Simple 2x2 RGB Image")
plt.show()
