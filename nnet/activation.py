import numpy as np


def relu(results):
    return results * (results > 0)


def relu_prime(results):
    return results > 0


def leaky_relu(results):
    greater = (results > 0) * results
    lesser = (results <= 0) * results * 0.0001
    return greater + lesser


def leaky_relu_prime(results):
    greater = results > 0
    lesser = 0 >= results
    lesser *= 0.0001
    return greater + lesser


def sigmoid(results):
    return 1 / (1 + np.exp(-results))


def sigmoid_prime(results):
    return np.multiply((1 - sigmoid(results)), sigmoid(results))


def tanh(results):
    return 2 * sigmoid(results) - 1

def tanh_prime(results):
    return 1 - tanh(results) ** 2


relu.prime = relu_prime
leaky_relu.prime = leaky_relu_prime
sigmoid.prime = sigmoid_prime
tanh.prime = tanh_prime
