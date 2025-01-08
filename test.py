import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Perform a basic tensor operation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = tf.matmul(a, b)

print("Result of matrix multiplication:\n", c)
