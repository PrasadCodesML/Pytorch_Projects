# Initially in 2002 torch was introduced where we can do tensor based operations on GPU but it was in Lua language and 
# then torch capability was coded in Python and then Pytorch was created by Facebook

# Tensorflow -> Google

# The calculations of tensors are stored in form of computation graph in Pytorch there are 
# dynamic computation graph -> The graph can change during runtime due to which experimentation and debugging becomes easy

# Core Features of PyTorch

# 1. Tensor Computations
# 2. GPU Acceleration
# 3. Dynamic Computation Graph
# 4. Automatic Differentiation
# 5. Distributed Training
# 6. Interoperability with other libraries

# Pytorch Vs Tensorflow
# 1. Programming Language
    # Pytorch: Python centric
    # Tensorflow: multi-language support

# 2. Ease of Use
    # Pytorch -> It is more Intuitive and easy to use
    # Tensorflow -> It has a more verbose syntax
    # Pytorch Wins

# 3. Deployment and Production
    # Pytorch -> offers TorchScript for model serialization, Pytorch mobile supports mobile deployment, growing support for production environments
    # Tensorflow -> Strong production support with Tensorflow serving, tensorflow lite and tensorflow.js more mature tools
    # Tensorflow Wins -> More mature and comprehensive deployment options

# 4. Performance 
    # Pytorch -> 
    # Tensorflow ->
    # Tie

# 5. Community and Ecosystem
    # Depends

# 6. High level API
    # Pytorch -> pytorch lightning, fast.ai
    # Tensorflow -> Keras
    # Tensorflow Wins -> Keras provides more established and user friendly high level API

# 7. Mobile and embedded deployment 
    # Pytorch -> supports mobile deployment
    # Tensorflow -> supports mobile deployment, and embedded deployment and tensorflow.js for web deployment
    # Tensorflow Wins -> More mature and versatile deployment

# 8. Learning Curve -> Pytorch wins -> More beginner friendly

# 9. Interoperability -> Pytorch wins -> Better integration with Python ecosystem

# 10. Customization -> Pytorch wins -> Greater customizability and flexibility

# 11. Deployment 
    # Pytorch -> TorchServe 
    # Tensorflow -> Tensorflow serving, Tensorflow Extended (TFX)

# 12. Parallelism and Distributed Training 
    # Pytorch -> torch.distributed
    # Tensorflow -> tf.distributed.Strategy
    # Tensorflow wins

# 13. Model Zoo and Pre-trained Models 
    # Pytorch -> Access via TorchVision and Hugging Face
    # Tensorflow -> Access via TensorFlow Hub (Kaggle models) and TensorFlow Model Garden
    # Tie

# Core Pytorch Modules
# 1. torch -> The core module providing multidimensional arrays (tensors) and mathematical operations on them
# 2. torch.autograd -> Automatic Differentiation engine that records operations on tensors to compute gradients for optimization
# 3. torch.nn -> Provides a neural networks library, including layers, activations, loss functions and utilizes to build deep learning models
# 4. torch.optim -> Provides optimization algorithms (optimizers) like SGD, RMSprop, Adam, etc.
# 5. torch.utils.data -> Provides utilities for loading and transforming datasets like Dataset and DataLoader
# 6. torch.jit -> Supports Just In Time (JIT) compilation and TorchScript for optimizing models and enabling deployment without Python Dependencies
# 7. torch.distributed -> Provides support for distributed training across multiple GPUs or TPUs
# 8. torch.cuda -> Interfaces with NVIDIA CUDA to enable GPU acceleration for tensor computation and model training
# 9. torch.backends -> Contains settings and allow control over backend libraries like cuDNN, MKL and other for performance tuning
# 10. torch.multiprocessing -> Utilizes for parallelism using multiprocessing, similar to Python's multiprocessing module but with support for CUDA tensors
# 11. torch.quantization -> Tools for model quantization to reduce model size and improve inference speed, especially on edge devices
# 12. torch.onnx -> Supports exporting PyTorch models to the ONNX (Open Neural Network Exchange) format for interoperability with other frameworks and deployment

# Pytorch Domain Libraries
# 1. torchvision -> Provides datasets, model architectures, and image transformations for computer vision tasks
# 2. torchtext -> Tools and datasets for NLP, including data preprocessing and vocabulary management
# 3. torchaudio -> Utilities for audio processing tasks, including i/o, transforms, and pretrained models for speech recognition 
# 4. torcharrow -> A library for accelerated data loading and preprocessing, especially for tabular and time series data (experimental)
# 5. torchserve -> A Pytorch model serving library that makes it easy to deploy trained models at scale in production environments
# 6. pytorch_lightning -> A lightweight wrapper for Pytorch that simplifies the training loop and reduces boilerplate code, enabling scalable and reproducible models

# Popular Pytorch ecosystem libraries
# Hugging face transformers, fastai, PyTorch Geometric, TorchMetrics, TorchElastic, optuna (Hyperparameter tuning), 
# Catalyst, Ignite, AllenNLP, Skorch, Pytorch Forecasting, Tensorboard for Pytorch