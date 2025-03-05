# What are tensors?
# Tensors is nothing but a data structure -> Tensor is a specialized multi-dimensional array designed for mathematical and computational efficiency

# O dimensional tensor (Scalar) -> A single number
# Represents a single value, often used for simple metrics or constants eg. 5.0

# 1 dimensional tensor (Vector) -> A list of numbers arranged in a single row or column
# eg. embedding vector is a vector

# 2 dimensional tensor (Matrix) -> A 2D array of numbers arranged in rows and columns
# eg. grey scale image is a 2d tensor

# 3 dimensional tensor (3D Tensor) -> A 3D array of numbers arranged in 3 dimensions (length, width, height)
# eg. RGB image is a 3D tensor

# 4 dimensional tensor (4D Tensor) -> A 4D array of numbers arranged in 4 dimensions (batch size, length, width, height)
# eg. Batches of RGB image are a 4D tensor

# 5 dimensional tensor (5D Tensor) -> A 5D array of numbers arranged in 5 dimensions (batch size, length, width, height, channels)
# eg. batch of RGB video clips is a 5D tensor (since each video is a 4d tensor)

# Why are tensors useful ?
# 1. Mathematical Operations -> Tensors enable efficient mathematical computations necessary for neural network
# 2. Representation of Real world data -> Data like images, audio, videos and text can be represented as tensors
# 3. Efficient computations -> Tensors are optimized for hardware optimization allowing computations on GPUs or TPUs which are crucial for training deep learning models

# Where are tensors used in Deep Learning?
# 1. Data Storage -> Training data (images, text, etc) is stored in tensors
# 2. Weights and Biases -> The learnable parameters of a neural network(weights, biases) are stored in tensors
# 3. Matrix Operations -> Neural networks involve operations like matrix multiplication, dot products and broadcasting - all performed using tensors
# 4. Training Process -> During forward passes, tensors flow through the neural networks, Gradients, represented as tensors are calculated during the backward pass



