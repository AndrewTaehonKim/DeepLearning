import numpy as np
import pytest
from deep_learning_modules.node import Node
import deep_learning_modules.operations as op

@pytest.fixture
def sample_data():
    X = np.array([
        [1, 1],
        [2, -1]
    ])
    y = np.array([
        [1, 10]
    ])
    y_hat = np.array([
        [2, 8]
    ])
    return X, y, y_hat

@pytest.fixture
def initial_weights(sample_data):
    X, _, _ = sample_data
    w = np.full(shape=(np.shape(X)[1], 1), fill_value=2)
    return w
@pytest.fixture
def initial_bias():
    return 1

def test_node_initialization(initial_weights, initial_bias):
    test_node = Node(
        loss_func_derivative=op.MSE_derivative,
        activation_func=op.ReLU,
        activation_derivative=op.ReLU_derivative,
        transform_func=op.quadratic_transform,
        transform_derivative=op.quadratic_transform_derivative,
        transform_params=initial_weights,
        b=initial_bias,
        transform_derivativeb=
    )