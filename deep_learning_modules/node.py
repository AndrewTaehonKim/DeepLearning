# This class defines a node in a NN
# This node is flexible and can compute any function by passing in the transform
# Typically each node will be a product of the weight, where ws
import numpy as np
from typing import Callable

class Node:
    # initialize a node object that takes in a transform function (f(inputs, weights))
    def __init__(
            self,
            loss_func_derivative: Callable,
            activation_func: Callable,
            activation_derivative: Callable,
            transform_func: Callable,
            transform_derivative: Callable,
            transform_params: np.ndarray,
            b,
            transform_derivativeb: Callable,
            layer_id:int=None,
            layer_type:str=None,
            edge_list=None
        ):
        self.loss_func_derivative = loss_func_derivative
        self.activation = activation_func
        self.activation_derivative = activation_derivative
        self.transform = transform_func
        self.transform_derivative = transform_derivative
        self.transform_params = transform_params
        self.b = b
        self.transform_derivativeb = transform_derivativeb
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.edge_list = edge_list

    # define a method to calculate the output for the node
    def get_value(self, xs:np.ndarray):
        self.value = self.transform(xs, self.transform_params)
        assert type(self.value) == np.ndarray
        return self.value
    
    # define a method to get the value after activation
    def forward_prop(self):
        assert type(self.value) == np.ndarray
        return self.activation_func(self.value)
    
    # define a method to calculate the derivative, where params are a list of lists of data that the node needs to calculate the derivative
    def get_gradient(self, params):
        dL = self.loss_func_derivative(params)
        dL_dactiv = np.multiply(dL, self.activation_derivative(params))
        # calculate derivative of loss with respect to weights
        dactiv_dtransform = np.multiply(dL_dactiv, self.transform_derivative(params))
        dw = np.average(dactiv_dtransform, axis=2)
        # calculate derivative of loss with respect to bias
        dactiv_dtransformb = np.multiply(dL_dactiv, self.transform_derivativeb(params))
        db = np.average(dactiv_dtransformb, axis=2)
        return -dw, -db
    
    # define a method to return the new set of weights and biases to use in the next forward step
    def backward_prop(self, alpha, params):
        gradw, gradb = self.get_gradient(params)
        self.transform_params += alpha*gradw
        self.b += alpha*gradb
        return self.transform_params, self.b


