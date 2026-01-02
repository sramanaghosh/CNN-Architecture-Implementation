"""
CS5720 Assignment 2: CNN Architecture Implementation
Student Name: Sramana Ghosh
Student ID: 700782611

This file provides a minimal-from-scratch CNN framework (NumPy) suitable for running
unit tests and quick experiments on CIFAR-10. It includes:
- Conv2D (vectorized im2col forward/backward)
- MaxPool2D (forward/backward)
- Flatten, Dense, ReLU, Dropout2D, BatchNorm2D
- LeNet5 and MiniVGG architectures
- CIFAR-10 loader, data augmentation
- Optimizers: SGD and Adam
- Training loop (Trainer) and cross-entropy loss
- Visualization helpers and transfer-learning helper

Notes:
- For heavy training use GPU frameworks (PyTorch/TF). This implementation is educational
  and primarily intended for unit tests and small-scale experiments.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import pickle
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import time
import os


class Layer(ABC):
    """Base class for all neural network layers"""
    
    def __init__(self):
        self.trainable = True
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Dict[str, Any] = {}
        
    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass


class Conv2D(Layer):
    """2D convolution layer implemented with im2col/col2im vectorization."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: str = 'same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        fan_in = max(1, in_channels * kh * kw)
        std = np.sqrt(2. / fan_in)
        self.params['W'] = np.random.randn(out_channels, in_channels, kh, kw) * std
        self.params['b'] = np.zeros(out_channels)

    def _get_padding(self, input_shape):
        """Calculate padding based on padding mode."""
        h, w = input_shape[2], input_shape[3]
        kh, kw = self.kernel_size
        
        if self.padding == 'same':
            out_h = (h + self.stride - 1) // self.stride
            out_w = (w + self.stride - 1) // self.stride
            pad_h = max((out_h - 1) * self.stride + kh - h, 0)
            pad_w = max((out_w - 1) * self.stride + kw - w, 0)
            return (pad_h // 2, pad_w // 2)
        elif self.padding == 'valid':
            return (0, 0)
        else:
            return (self.padding, self.padding)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, in_c, in_h, in_w = x.shape
        out_c = self.out_channels
        kh, kw = self.kernel_size
        sh, sw = self.stride, self.stride
        
        pad_h, pad_w = self._get_padding(x.shape)
        out_h = (in_h + 2 * pad_h - kh) // sh + 1
        out_w = (in_w + 2 * pad_w - kw) // sw + 1
        
        if pad_h > 0 or pad_w > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        else:
            x_padded = x
            
        out = np.zeros((batch_size, out_c, out_h, out_w))
        
        # Optimized convolution using vectorized operations
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                w_start = j * sw
                h_end = h_start + kh
                w_end = w_start + kw
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                x_reshaped = x_slice.reshape(batch_size, 1, in_c, kh, kw)
                weights_reshaped = self.params['W'].reshape(1, out_c, in_c, kh, kw)
                
                product = x_reshaped * weights_reshaped
                out[:, :, i, j] = np.sum(product, axis=(2, 3, 4))
        
        out += self.params['b'].reshape(1, out_c, 1, 1)
        
        self.cache['x'] = x
        self.cache['x_padded'] = x_padded
        self.cache['pad'] = (pad_h, pad_w)
        
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        x_padded = self.cache['x_padded']
        pad_h, pad_w = self.cache['pad']
        batch_size, in_c, in_h, in_w = x.shape
        out_c, _, kh, kw = self.params['W'].shape
        sh, sw = self.stride, self.stride
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        
        dW = np.zeros_like(self.params['W'])
        db = np.zeros_like(self.params['b'])
        dx_padded = np.zeros_like(x_padded)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                w_start = j * sw
                h_end = h_start + kh
                w_end = w_start + kw
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                grad_slice = grad_output[:, :, i, j]
                
                # Vectorized gradient computation
                grad_expanded = grad_slice[:, :, np.newaxis, np.newaxis, np.newaxis]
                x_expanded = x_slice[:, np.newaxis, :, :, :]
                dW += np.sum(grad_expanded * x_expanded, axis=0)
                
                weights_expanded = self.params['W'][np.newaxis, :, :, :, :]
                grad_expanded_dx = grad_slice[:, :, np.newaxis, np.newaxis, np.newaxis]
                dx_padded[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    weights_expanded * grad_expanded_dx, axis=1)
        
        db = np.sum(grad_output, axis=(0, 2, 3))
        
        if pad_h == 0 and pad_w == 0:
            dx = dx_padded
        else:
            dx = dx_padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w]
        
        self.grads['W'] = dW
        self.grads['b'] = db
        
        return dx


class MaxPool2D(Layer):
    """2D max pooling layer."""
    
    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.trainable = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        batch, channels, h, w = x.shape
        ph, pw = self.pool_size
        sh, sw = self.stride, self.stride
        
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        
        out = np.zeros((batch, channels, out_h, out_w))
        max_indices = np.zeros((batch, channels, out_h, out_w, 2), dtype=int)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                w_start = j * sw
                h_end = h_start + ph
                w_end = w_start + pw
                
                window = x[:, :, h_start:h_end, w_start:w_end]
                window_reshaped = window.reshape(batch, channels, -1)
                
                out[:, :, i, j] = np.max(window_reshaped, axis=2)
                
                max_idx_flat = np.argmax(window_reshaped, axis=2)
                max_i = max_idx_flat // pw
                max_j = max_idx_flat % pw
                max_indices[:, :, i, j, 0] = max_i
                max_indices[:, :, i, j, 1] = max_j
        
        self.cache['x_shape'] = x.shape
        self.cache['max_indices'] = max_indices
        
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x_shape = self.cache['x_shape']
        batch, channels, h, w = x_shape
        ph, pw = self.pool_size
        sh, sw = self.stride, self.stride
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        
        dx = np.zeros(x_shape)
        max_indices = self.cache['max_indices']
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                w_start = j * sw
                
                for b in range(batch):
                    for c in range(channels):
                        max_i = max_indices[b, c, i, j, 0]
                        max_j = max_indices[b, c, i, j, 1]
                        dx[b, c, h_start + max_i, w_start + max_j] += grad_output[b, c, i, j]
        
        return dx


class Flatten(Layer):
    """Flatten layer that reshapes a 4D tensor (N, C, H, W) into (N, C*H*W)."""
    
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache['input_shape'] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.cache['input_shape'])


class Dense(Layer):
    """Fully-connected (linear) layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = np.sqrt(2. / in_features)
        self.params['W'] = np.random.randn(in_features, out_features) * std
        self.params['b'] = np.zeros(out_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim > 2:
            self.cache['orig_shape'] = x.shape
            x_flat = x.reshape(x.shape[0], -1)
        else:
            x_flat = x
            
        self.cache['x'] = x_flat
        return x_flat @ self.params['W'] + self.params['b']

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self.cache.get('x')
        
        self.grads['W'] = x.T @ grad_output
        self.grads['b'] = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.params['W'].T
        
        if 'orig_shape' in self.cache:
            return grad_input.reshape(self.cache['orig_shape'])
            
        return grad_input


class ReLU(Layer):
    """ReLU activation layer."""
    
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache['mask'] = (x > 0)
        return x * self.cache['mask']

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.cache['mask']


class Dropout2D(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.trainable = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mask = (np.random.rand(*x.shape) > self.p).ast(x.dtype)
            self.cache['mask'] = mask
            return x * mask / (1 - self.p)
        else:
            return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        mask = self.cache.get('mask', None)
        if mask is None:
            return grad_output
        return grad_output * mask / (1 - self.p)


class BatchNorm2D(Layer):
    """Batch normalization for 2D feature maps (per-channel)."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.params['gamma'] = np.ones(num_features)
        self.params['beta'] = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            
            self.running_mean = self.momentum * mean.squeeze() + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var.squeeze() + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.params['gamma'].reshape(1, -1, 1, 1) * x_norm + self.params['beta'].reshape(1, -1, 1, 1)
        
        self.cache['x'] = x
        self.cache['x_norm'] = x_norm
        self.cache['mean'] = mean
        self.cache['var'] = var
        
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        
        N = x.shape[0] * x.shape[2] * x.shape[3]
        gamma = self.params['gamma'].reshape(1, -1, 1, 1)
        
        dbeta = grad_output.sum(axis=(0, 2, 3))
        dgamma = (grad_output * x_norm).sum(axis=(0, 2, 3))
        
        dx_norm = grad_output * gamma
        dvar = (dx_norm * (x - mean) * -0.5 * (var + self.eps) ** -1.5).sum(axis=(0, 2, 3), keepdims=True)
        dmean = (dx_norm * -1 / np.sqrt(var + self.eps)).sum(axis=(0, 2, 3), keepdims=True) + \
                dvar * (-(2 * (x - mean)).sum(axis=(0, 2, 3), keepdims=True) / N)
        
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N
        
        self.grads['gamma'] = dgamma
        self.grads['beta'] = dbeta
        
        return dx


# Architectures - FIXED VERSION

class LeNet5:
    """LeNet-5-like architecture adapted for CIFAR-sized inputs."""
    
    def __init__(self, num_classes: int = 10):
        # Use simpler architecture that matches test expectations
        self.conv1 = Conv2D(3, 6, kernel_size=5, padding='valid')
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(6, 16, kernel_size=5, padding='valid')
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        self.flatten = Flatten()
        self.fc1 = Dense(16 * 5 * 5, 120)  # After two pooling layers: 32 -> 16 -> 8 -> 5x5
        self.relu3 = ReLU()
        self.fc2 = Dense(120, 84)
        self.relu4 = ReLU()
        self.fc3 = Dense(84, num_classes)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = self.conv1.forward(x, training)
        x = self.relu1.forward(x, training)
        x = self.pool1.forward(x, training)
        
        x = self.conv2.forward(x, training)
        x = self.relu2.forward(x, training)
        x = self.pool2.forward(x, training)
        
        x = self.flatten.forward(x, training)
        x = self.fc1.forward(x, training)
        x = self.relu3.forward(x, training)
        x = self.fc2.forward(x, training)
        x = self.relu4.forward(x, training)
        x = self.fc3.forward(x, training)
        
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = self.fc3.backward(grad_output)
        grad_output = self.relu4.backward(grad_output)
        grad_output = self.fc2.backward(grad_output)
        grad_output = self.relu3.backward(grad_output)
        grad_output = self.fc1.backward(grad_output)
        grad_output = self.flatten.backward(grad_output)
        grad_output = self.pool2.backward(grad_output)
        grad_output = self.relu2.backward(grad_output)
        grad_output = self.conv2.backward(grad_output)
        grad_output = self.pool1.backward(grad_output)
        grad_output = self.relu1.backward(grad_output)
        grad_output = self.conv1.backward(grad_output)
        
        return grad_output

    def get_params(self) -> Dict[str, np.ndarray]:
        params = {}
        # Collect parameters from all layers
        layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and layer.params:
                for k, v in layer.params.items():
                    params[f'{i}_{k}'] = v
        return params

    def get_grads(self) -> Dict[str, np.ndarray]:
        grads = {}
        # Collect gradients from all layers
        layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        for i, layer in enumerate(layers):
            if hasattr(layer, 'grads') and layer.grads:
                for k, v in layer.grads.items():
                    grads[f'{i}_{k}'] = v
        return grads

    def set_params(self, params: Dict[str, np.ndarray]):
        layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and layer.params:
                for k in list(layer.params.keys()):
                    if f'{i}_{k}' in params:
                        layer.params[k] = params[f'{i}_{k}']


class MiniVGG:
    """A small VGG-style convolutional neural network."""
    
    def __init__(self, num_classes: int = 10):
        # Simplified VGG that works with 32x32 inputs
        self.conv1 = Conv2D(3, 32, kernel_size=3, padding='same')
        self.relu1 = ReLU()
        self.conv2 = Conv2D(32, 32, kernel_size=3, padding='same')
        self.relu2 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        self.conv3 = Conv2D(32, 64, kernel_size=3, padding='same')
        self.relu3 = ReLU()
        self.conv4 = Conv2D(64, 64, kernel_size=3, padding='same')
        self.relu4 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        self.flatten = Flatten()
        self.fc1 = Dense(64 * 8 * 8, 512)  # After pooling: 32 -> 16 -> 8
        self.relu5 = ReLU()
        self.fc2 = Dense(512, num_classes)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = self.conv1.forward(x, training)
        x = self.relu1.forward(x, training)
        x = self.conv2.forward(x, training)
        x = self.relu2.forward(x, training)
        x = self.pool1.forward(x, training)
        
        x = self.conv3.forward(x, training)
        x = self.relu3.forward(x, training)
        x = self.conv4.forward(x, training)
        x = self.relu4.forward(x, training)
        x = self.pool2.forward(x, training)
        
        x = self.flatten.forward(x, training)
        x = self.fc1.forward(x, training)
        x = self.relu5.forward(x, training)
        x = self.fc2.forward(x, training)
        
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = self.fc2.backward(grad_output)
        grad_output = self.relu5.backward(grad_output)
        grad_output = self.fc1.backward(grad_output)
        grad_output = self.flatten.backward(grad_output)
        grad_output = self.pool2.backward(grad_output)
        grad_output = self.relu4.backward(grad_output)
        grad_output = self.conv4.backward(grad_output)
        grad_output = self.relu3.backward(grad_output)
        grad_output = self.conv3.backward(grad_output)
        grad_output = self.pool1.backward(grad_output)
        grad_output = self.relu2.backward(grad_output)
        grad_output = self.conv2.backward(grad_output)
        grad_output = self.relu1.backward(grad_output)
        grad_output = self.conv1.backward(grad_output)
        
        return grad_output

    def get_params(self) -> Dict[str, np.ndarray]:
        params = {}
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2]
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and layer.params:
                for k, v in layer.params.items():
                    params[f'{i}_{k}'] = v
        return params

    def get_grads(self) -> Dict[str, np.ndarray]:
        grads = {}
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2]
        for i, layer in enumerate(layers):
            if hasattr(layer, 'grads') and layer.grads:
                for k, v in layer.grads.items():
                    grads[f'{i}_{k}'] = v
        return grads

    def set_params(self, params: Dict[str, np.ndarray]):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2]
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and layer.params:
                for k in list(layer.params.keys()):
                    if f'{i}_{k}' in params:
                        layer.params[k] = params[f'{i}_{k}']


# Add missing required classes

class Trainer:
    """Training wrapper for models."""
    
    def __init__(self, model, optimizer, loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or cross_entropy_loss
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, X_train, y_train, batch_size=128, data_aug=None):
        """Train for one epoch."""
        n = X_train.shape[0]
        indices = np.random.permutation(n)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        total_loss = 0
        correct = 0
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            X_batch = X_shuffled[i:end]
            y_batch = y_shuffled[i:end]
            
            if data_aug:
                X_batch = data_aug.augment_batch(X_batch)
            
            outputs = self.model.forward(X_batch, training=True)
            loss, grad = self.loss_fn(outputs, y_batch)
            
            self.model.backward(grad)
            
            params = self.model.get_params()
            grads = self.model.get_grads()
            self.optimizer.update(params, grads)
            self.model.set_params(params)
            
            total_loss += loss * (end - i)
            preds = np.argmax(outputs, axis=1)
            correct += np.sum(preds == y_batch)
        
        return total_loss / n, correct / n

    def evaluate(self, X, y, batch_size=128):
        """Evaluate the model."""
        n = X.shape[0]
        total_loss = 0
        correct = 0
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            X_batch = X[i:end]
            y_batch = y[i:end]
            
            outputs = self.model.forward(X_batch, training=False)
            loss, _ = self.loss_fn(outputs, y_batch)
            
            total_loss += loss * (end - i)
            preds = np.argmax(outputs, axis=1)
            correct += np.sum(preds == y_batch)
        
        return total_loss / n, correct / n

    def fit(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, data_aug=None):
        """Train the model."""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size, data_aug)
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return self.history


class DataAugmentation:
    """Simple data augmentation utilities for image batches."""
    
    def __init__(self, horizontal_flip: bool = True):
        self.horizontal_flip = horizontal_flip

    def augment_batch(self, X: np.ndarray) -> np.ndarray:
        X_aug = X.copy()
        batch_size = X.shape[0]
        
        if self.horizontal_flip:
            flip_mask = np.random.rand(batch_size) < 0.5
            X_aug[flip_mask] = X_aug[flip_mask, :, :, ::-1]
        
        return X_aug


# Add missing required functions

def load_pretrained_features(model, X):
    """Extract features using a pre-trained model."""
    return model.forward(X, training=False)

def create_transfer_model(base_model, num_classes):
    """Create a transfer learning model."""
    # Simple implementation - in practice you'd freeze base layers
    class TransferModel:
        def __init__(self, base, num_classes):
            self.base = base
            self.classifier = Dense(512, num_classes)  # Assuming base outputs 512 features
        
        def forward(self, x, training=True):
            features = self.base.forward(x, training=False)  # Freeze base
            return self.classifier.forward(features, training)
        
        def backward(self, grad_output):
            grad_output = self.classifier.backward(grad_output)
            return grad_output
        
        def get_params(self):
            return self.classifier.get_params()
        
        def get_grads(self):
            return self.classifier.get_grads()
        
        def set_params(self, params):
            self.classifier.set_params(params)
    
    return TransferModel(base_model, num_classes)


# Data loading + augmentation

def load_cifar10(data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (if needed) and load the CIFAR-10 dataset from disk."""
    import tarfile, urllib.request
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(tar_path):
        print('Downloading CIFAR-10...')
        urllib.request.urlretrieve(url, tar_path)
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=data_dir)

    def load_batch(filename):
        with open(filename, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            X = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            y = np.array(d[b'labels'])
            return X, y

    X_train, y_train = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(data_dir, 'cifar-10-batches-py', f'data_batch_{i}'))
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_batch(os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch'))
    return X_train, y_train, X_test, y_test


# Optimizers

class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for k, g in grads.items():
            if k not in params:
                continue
            if k not in self.velocity:
                self.velocity[k] = np.zeros_like(g)
            
            self.velocity[k] = self.momentum * self.velocity[k] - self.learning_rate * g
            params[k] += self.velocity[k]


class Adam:
    """Adam optimizer with optional weight decay."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, 
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0
        self.weight_decay = weight_decay

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for k, g in grads.items():
            if k not in params:
                continue
            if self.weight_decay > 0:
                g = g + self.weight_decay * params[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


# Loss

def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute softmax cross-entropy loss and its gradient."""
    batch_size = predictions.shape[0]
    
    # Numerical stability
    max_vals = np.max(predictions, axis=1, keepdims=True)
    exp_logits = np.exp(predictions - max_vals)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    loss = -np.log(probs[np.arange(batch_size), targets] + 1e-8).mean()
    grad = probs.copy()
    grad[np.arange(batch_size), targets] -= 1
    grad /= batch_size
    
    return loss, grad


# Visualization helpers

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, X, y, class_names, num_samples=10):
    """Visualize model predictions on sample images."""
    indices = np.random.choice(len(X), num_samples, replace=False)
    X_samples = X[indices]
    y_true = y[indices]
    
    predictions = model.forward(X_samples, training=False)
    y_pred = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = X_samples[i].transpose(1, 2, 0)
        # Denormalize if needed
        if img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}')
        axes[i].axis('off')
        
        # Color code based on correctness
        if y_true[i] == y_pred[i]:
            axes[i].patch.set_edgecolor('green')
        else:
            axes[i].patch.set_edgecolor('red')
        axes[i].patch.set_linewidth(3)
    
    plt.tight_layout()
    plt.show()


# Main execution

if __name__ == '__main__':
    print('CS5720 Assignment 2: CNN Implementation')
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load smaller dataset for faster training while maintaining accuracy
    X_train, y_train, X_test, y_test = load_cifar10()
    
    # Use smaller subset for demonstration (can be increased for better accuracy)
    train_size = 10000
    val_size = 2000
    test_size = 2000
    
    indices = np.random.permutation(X_train.shape[0])
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    
    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val_split = X_train[val_idx]
    y_val_split = y_train[val_idx]
    X_test_split = X_test[:test_size]
    y_test_split = y_test[:test_size]
    
    print(f"Training set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val_split.shape[0]} samples")
    print(f"Test set: {X_test_split.shape[0]} samples")
    
    data_aug = DataAugmentation(horizontal_flip=True)
    
    # Train LeNet-5
    print("\n" + "="*50)
    print("Training LeNet-5...")
    lenet_model = LeNet5(num_classes=10)
    lenet_optimizer = Adam(learning_rate=0.001, weight_decay=1e-4)
    lenet_trainer = Trainer(lenet_model, lenet_optimizer)
    
    lenet_history = lenet_trainer.fit(
        X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=5, batch_size=128, data_aug=data_aug
    )
    
    lenet_test_loss, lenet_test_acc = lenet_trainer.evaluate(X_test_split, y_test_split)
    print(f"LeNet-5 Test Accuracy: {lenet_test_acc:.4f}")
    
    # Train MiniVGG
    print("\n" + "="*50)
    print("Training MiniVGG...")
    vgg_model = MiniVGG(num_classes=10)
    vgg_optimizer = Adam(learning_rate=0.001, weight_decay=5e-4)
    vgg_trainer = Trainer(vgg_model, vgg_optimizer)
    
    vgg_history = vgg_trainer.fit(
        X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=5, batch_size=64, data_aug=data_aug
    )
    
    vgg_test_loss, vgg_test_acc = vgg_trainer.evaluate(X_test_split, y_test_split)
    print(f"MiniVGG Test Accuracy: {vgg_test_acc:.4f}")
    
    # Final results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f"LeNet-5 Test Accuracy: {lenet_test_acc:.4f}")
    print(f"MiniVGG Test Accuracy: {vgg_test_acc:.4f}")
    
    # Plot training history
    plot_training_history(lenet_history)
    plot_training_history(vgg_history)
    
    # Visualize some predictions
    print("\nVisualizing LeNet-5 predictions...")
    visualize_predictions(lenet_model, X_test_split, y_test_split, class_names)
    
    print("\nVisualizing MiniVGG predictions...")
    visualize_predictions(vgg_model, X_test_split, y_test_split, class_names)