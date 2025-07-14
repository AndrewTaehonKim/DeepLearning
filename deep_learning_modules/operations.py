# This file contains function definitions for fundamental operations
import numpy as np

### LOSS FUNCTIONS ###
# loss function - MSE
def MSE(y, y_hat):
    return np.mean(np.square(y-y_hat))

### LOSS FUNCTION DERIVATIVES ###
# loss function derivative - MSE
def MSE_derivative(y, y_hat):
    return y - y_hat

### ACTIVATION FUNCTIONS ###
# activation_function - ReLU
def ReLU(x):
    return np.maximum(0, x)

### ACTIVATION DERIVATIVES ###
# activation function derivative - ReLU
def ReLU_derivative(x):
    return np.where(ReLU(x) > 0, 1, ReLU(x))

### TRANSFORMATIONS ###
# transform - linear wx
def linear_transform(w, x):
    return np.dot(w.T, x)

# transform - quadratic transform wx^2
def quadratic_transform(w, x):
    return np.dot(w.T, np.square(x))

### TRANSFORMATION DERIVATIVES
# transformation derivative - quadratic terms w
def linear_transform_derivative(w, x):
    return w

# transformation derivative - quadratic terms 2wx
def quadratic_transform_derivative(w, x):
    return 2*np.multiply(w.T,x)

# transform derivative for bias
def transform_derivativeb(w,x):
    pass