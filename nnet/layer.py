import numpy as np

import nnet


class NeuralNetworkLayer(object):
    """
    A generic layer class meant to be derived, not instantiated.
    """

    def __init__(self, kwargs):
        super(NeuralNetworkLayer, self).__init__()

        self.output_size = None
        self.current_batch_X = None
        self.current_batch_Y = None
        self.pass_type = None

        try:
            self.output_size = kwargs["output_size"]
            self.pass_type = kwargs["pass_type"].split('|')
        except KeyError:
            pass

        if self.pass_type is None:
            self.pass_type = kwargs["pass_type"]

    def receive_current_batch_info(self, X, Y):
        self.current_batch_X = X
        self.current_batch_Y = Y

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

    def update_parameters(self, learning_rate, regularization_strength):
        pass


class FullyConnectedLayer(NeuralNetworkLayer):
    """
    """

    def __init__(self, initialization_type="xavier", **kwargs):
        super(FullyConnectedLayer, self).__init__(kwargs)
        self.W = None
        self.dW = None
        self.local_input_gradient = None
        self.local_weight_gradient = None
        self.initialization_type=initialization_type

    def initialize(self, input_size):
        """
        Initializes the weights of the fully connected layer.

        :param input_size: The size of the input to the layer.
        :return: None
        """
        if self.initialization_type == "xavier":
            self.W = np.random.randn(self.output_size, input_size) / np.sqrt(input_size)
        elif self.initialization_type == "xavier_relu":
            self.W = np.random.randn(self.output_size, input_size) / np.sqrt(input_size/2)
        elif self.initialization_type == "gaussian":
            self.W = np.random.randn(self.output_size, input_size).astype(np.float64) * 0.01
        else:
            print("Error: initialization Type not recognized")
            exit(1)

    def forward_pass(self, inputs):
        """
        Performs a forward pass at this layer, caches the local
        gradients wrt inputs and weights.

        :param inputs: The variables to multiply the weight with.
        :return: The dot product of this layer's weights and the inputs.
        """
        self.local_input_gradient = self.W.T
        self.local_weight_gradient = inputs.T
        return self.W.dot(inputs)

    def backward_pass(self, previous_gradient):
        """
        Performs a backward pass at this layer. Right multiplies the
        previous gradient with the cached weight gradient. Left multiplies
        the previous gradient with the cached input gradient.

        :param previous_gradient: The gradient coming into this layer.
        :return: Gradient of the loss wrt the forward input to this layer.
        """
        self.dW = np.dot(previous_gradient, self.local_weight_gradient)
        next_gradient = np.dot(self.local_input_gradient, previous_gradient)
        return next_gradient

    def update_parameters(self, learning_rate, regularization_strength):
        self.W += -learning_rate * (self.dW + regularization_strength * self.W)


class SoftmaxLayer(NeuralNetworkLayer):
    def __init__(self, **kwargs):
        super(SoftmaxLayer, self).__init__(kwargs)

    def initialize(self, **kwargs):
        """
        Initializes the softmax layer.

        :param input_size: The size of the input to the layer.
        :return: None
        """
        self.output_size = kwargs['input_size']

    def forward_pass(self, inputs):
        return nnet.loss.softmax(inputs)


class LossLayer(NeuralNetworkLayer):
    def __init__(self, loss_function, **kwargs):
        super(LossLayer, self).__init__(kwargs)
        self.output_size = 1
        self.local_gradient = None
        if loss_function == "softmax":
            self.loss_function = nnet.loss.softmax

    def forward_pass(self, inputs):
        loss, self.local_gradient = self.loss_function.gradient(inputs, self.current_batch_Y)
        return loss

    def backward_pass(self, previous_gradient):
        # since this is the loss layer, it shouldn't have a previous gradient.
        return self.local_gradient * previous_gradient


class ActivationLayer(NeuralNetworkLayer):
    """
    """
    def __init__(self, activation_function, **kwargs):
        super(ActivationLayer, self).__init__(kwargs)

        self.local_gradient = None

        if activation_function == "relu":
            self.activation_function = nnet.activation.relu
        elif activation_function == "leaky_relu":
            self.activation_function = nnet.activation.leaky_relu
        elif activation_function == "sigmoid":
            self.activation_function = nnet.activation.sigmoid
        elif activation_function == "tanh":
            self.activation_function = nnet.activation.tanh

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
        self.local_gradient = self.activation_function.prime(inputs)
        return self.activation_function(inputs)

    def backward_pass(self, previous_gradient):
        """
        Performs element wise multiplication with the gradient flowing into this
        layer and the cached local gradient.

        :param previous_gradient: the gradient coming into this layer
        :return: The gradient of the loss function wrt this layer's activation function.
        """
        return self.local_gradient * previous_gradient
