import numpy as np


class PassType:
    TRAIN=0
    TEST=1


class NeuralNetworkLayer(object):
    """
    A generic layer class meant to be derived, not instantiated.
    """
    def __init__(self, pass_type=PassType.TRAIN):
        super(NeuralNetworkLayer, self).__init__()
        self.output_size = None
        self.pass_type = pass_type

    def initialize(self, **kwargs):
        """
        Initialization template. Should be called once before training.

        :return: None
        """
        pass

    def forward_pass(self, inputs):
        """
        A forward pass template

        :param inputs: the inputs to the layer
        :return: None
        """
        pass

    def backward_pass(self, previous_gradient):
        """
        A backward pass template

        :param previous_gradient: the gradient coming into this layer
        :return: None
        """
        pass


class FullyConnectedLayer(NeuralNetworkLayer):
    """
    """
    def __init__(self, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.output_size = output_size
        self.W = None

    def initialize(self, **kwargs):
        """
        Initializes the weights of the fully connected layer.

        :param input_size: The size of the input to the layer.
        :return: None
        """
        input_size = kwargs['input_size']
        self.W = np.random.randn(self.output_size, input_size) / np.sqrt(input_size)

    def forward_pass(self, inputs):
        """
        Performs a forward pass at this layer, caches the local
        gradients wrt inputs and weights.

        :param inputs: The variables to multiply the weight with.
        :return: The dot product of this layer's weights and the inputs.
        """
        self.input_gradient = self.W.T
        self.weight_gradient = inputs.T
        return self.W.dot(inputs)

    def backward_pass(self, previous_gradient):
        """
        Performs a backward pass at this layer. Right multiplies the
        previous gradient with the cached weight gradient. Left multiplies
        the previous gradient with the cached input gradient.

        :param previous_gradient: The gradient coming into this layer.
        :return: Gradient of the loss wrt the forward input to this layer.
        """
        self.dW = previous_gradient.dot(self.weight_gradient)
        dInput = self.input_gradient.dot(previous_gradient)
        return dInput


class ActivationLayer(object):
    """
    """
    def __init__(self, activation_function):
        super(ActivationLayer, self).__init__()
        self.output_size = None
        self.activation_function = activation_function


    def initialize(self, **kwargs):
        """
        Initializes the activation layer.

        :param input_size: The size of the input to the layer.
        :return: None
        """
        self.output_size = kwargs['input_size']

    def forward_pass(self, inputs):
        """
        Performs a forward pass at this layer. Simply passes the inputs through
        an activation function and caches it's local gradient.

        :param inputs: the inputs to the layer
        :return: None
        """
        self.cache = self.activation_function.prime(inputs)
        return self.activation_function(inputs)

    def backward_pass(self, previous_gradient):
        """
        Performs element wise multiplication with the gradient flowing into this
        layer and the cached local gradient.

        :param previous_gradient: the gradient coming into this layer
        :return: The gradient of the loss function wrt this layer's activation function.
        """
        return self.cache * previous_gradient


if __name__ == "__main__":
    nn = FullyConnectedLayer(10)
    nn.initialize(input_size=20, apple="fruit", celery="vegetables")
