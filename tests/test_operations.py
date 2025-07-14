# test the operations.py file
import numpy as np
import pytest

import deep_learning_modules.operations as op

@pytest.fixture
def sample_data():
    X = np.array([
        [1, 1],
        [2, -1]
    ])
    y = np.array([
        [1, -10]
    ])
    y_hat = np.array([
        [2, 8]
    ])
    return X, y, y_hat

@pytest.fixture
def sample_weights(sample_data):
    X, _, _ = sample_data
    w = np.full(shape=(np.shape(X)[1], 1), fill_value=2)
    return w


# TESTS 
def test_MSE(sample_data):
    _, y, y_hat = sample_data
    MSE = op.MSE(y, y_hat)
    expected_MSE = np.sum(np.square(y-y_hat))/2
    assert np.isclose(MSE, expected_MSE), "MSE computation failed"

def test_MSE_derivative(sample_data):
    _, y, y_hat = sample_data
    MSE_derivative = op.MSE_derivative(y, y_hat)
    expected_MSE_derivative = y - y_hat
    assert np.allclose(MSE_derivative, expected_MSE_derivative), "MSE derivative computation failed"

def test_ReLU(sample_data):
    _, y, _ = sample_data
    ReLU = op.ReLU(y)
    expected_ReLU = np.maximum(0, y)
    assert np.allclose(ReLU, expected_ReLU), "ReLU computation failed"

def test_ReLU_derivative(sample_data):
    _, y, _ = sample_data
    ReLU = op.ReLU_derivative(y)
    expected_ReLU = np.where(op.ReLU(y)>0, 1, op.ReLU(y))
    assert np.allclose(ReLU, expected_ReLU), "ReLU computation failed"

def test_linear_transform(sample_data, sample_weights):
    X, _, _ = sample_data
    w = sample_weights
    linear_transform = op.linear_transform(w, X)
    expected_linear_transform = w.T@X
    assert np.allclose(linear_transform, expected_linear_transform), "linear transformation failed"

def test_quadratic_transform(sample_data, sample_weights):
    X, _, _ = sample_data
    w = sample_weights
    quadratic_transform = op.quadratic_transform(w, X)
    expected_quadratic_transform = w.T@np.square(X)
    assert np.allclose(quadratic_transform, expected_quadratic_transform), "quadratic transformation failed"

def test_linear_transform_derivative(sample_data, sample_weights):
    X, _, _ = sample_data
    w = sample_weights
    linear_transform_derivative = op.linear_transform_derivative(w,X)
    expected_linear_transform_derivative = w
    assert np.allclose(linear_transform_derivative, expected_linear_transform_derivative), "linear transform derivative failed"

def test_quadratic_tansform_derivative(sample_data, sample_weights):
    X, _, _ = sample_data
    w = sample_weights
    quadratic_tansform_derivative = op.quadratic_transform_derivative(w,X)
    expected_quadratic_tansform_derivative = 2*w.T*X
    assert np.allclose(quadratic_tansform_derivative, expected_quadratic_tansform_derivative), "quadratic transform derivative failed"