import tensorflow as tf
import matplotlib.pyplot as plt

# Create a simple 2x2x2 image tensor
image_tensor = tf.constant([[[1, 0], [0, 1]],
                            [[1, 0], [0, 1]]], dtype=tf.float32)

print("Image Tensor:")
print(image_tensor)
print("Shape of the tensor:", image_tensor.shape)

# Adjust the tensor to have 3 channels by duplicating the channels to form an RGB image
image_tensor_3d = tf.concat([image_tensor[..., tf.newaxis]] * 3, axis=-1)

print("Image Tensor with 3 Channels:")
print(image_tensor_3d)
print("Shape of the tensor with 3 channels:", image_tensor_3d.shape)

# Display the image
plt.imshow(image_tensor_3d.numpy())
plt.title("Simple 2x2x3 Image Tensor")
plt.show()
