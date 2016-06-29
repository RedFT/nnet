def relu(inputs):
    return inputs * (inputs > 0)

def relu_prime(inputs):
    return inputs > 0

relu.prime = relu_prime
