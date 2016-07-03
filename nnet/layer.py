import numpy as np

import nnet


class NeuralNetworkLayer(object):
    """
    A generic layer class meant to be derived, not instantiated.
    """

    def __init__(self, output_size, pass_type="train"):
        """
        This is where output_size and pass_type will be handled

        :param output_size: The size of the output
        :param pass_type: The type of forward propogation. (Either "test", "train", or "test|train")

        """
        super(NeuralNetworkLayer, self).__init__()

        self.current_batch_X = None
        self.current_batch_Y = None
        self.pass_type = None
        self.output_size = output_size
        self.pass_type = pass_type.split('|')
        if self.pass_type is None:
            self.pass_type = pass_type

    def receive_current_batch_info(self, X, Y):
        """
        This method allows the layer to "know" what the input data and labels will be.

        :param X: The input data.
        :param Y: The input labels.

        """
        self.current_batch_X = X
        self.current_batch_Y = Y

    def initialize(self, input_size):
        """
        Initialization template. Should be called once before training.

        :param input_size: The size of the input of this layer.
        """
        pass

    def forward_pass(self, inputs):
        """
        A forward pass template

        :param inputs: the inputs to the layer

        """
        pass

    def backward_pass(self, previous_gradient):
        """
        A backward pass template

        :param previous_gradient: the gradient coming into this layer

        """
        pass

    def update_parameters(self, learning_rate, regularization_strength):
        """
        A template for updating parameters. FullyConnected Layers and Batch Normalization Layers will use this.

        :param learning_rate: The step size. Determines the amount to "move" in each dimension of the weights.
        :param regularization_strength: The value to multiply the weights by, to prevent overfitting.
        """
        pass


class FullyConnectedLayer(NeuralNetworkLayer):
    """
    During a forward pass, this layer performs a dot product operation on it's input with it's weights.

    During a backward pass, this layer performs a dot product operation on it's forward input transposed
    and the previous gradient.
    """

    def __init__(self, output_size, pass_type="train", initialization_type="xavier"):
        """
        Specify some properties of the layer.

        :param output_size: The size of this layer's output
        :param pass_type: The type of forward propogation. (Either "test", "train", or "test|train")
        :param initialization_type:

        """
        super(FullyConnectedLayer, self).__init__(output_size, pass_type)
        self.W = None
        self.dW = None
        self.local_input_gradient = None
        self.local_weight_gradient = None
        self.initialization_type = initialization_type

    def initialize(self, input_size):
        """
        Initializes the weights of the fully connected layer.

        :param input_size: The size of the input to the layer.

        """
        if self.initialization_type == "xavier":
            self.W = np.random.randn(self.output_size, input_size) / np.sqrt(input_size)
        elif self.initialization_type == "xavier_relu":
            self.W = np.random.randn(self.output_size, input_size) / np.sqrt(input_size / 2)
        elif self.initialization_type == "gaussian":
            self.W = np.random.randn(self.output_size, input_size).astype(np.float64) * 0.01
        else:
            print("Error: initialization Type '" + self.initialization_type + "' not recognized")
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
        """
        Updates this layer's weight using `learning_rate` and `regularization_strength`

        :param learning_rate: The step size. Determines the amount to "move" in each dimension of the weights.
        :param regularization_strength: The value to multiply the weights by, to prevent overfitting.

        """
        self.W += -learning_rate * (self.dW + regularization_strength * self.W)


class SoftmaxLayer(NeuralNetworkLayer):
    """
    During a forward pass, this simply computes the softmax probabilities
    on it's inputs.
    """

    def __init__(self, pass_type="test"):
        """
        Specify the pass_type
        :param pass_type: The type of forward propogation.
        """
        super(SoftmaxLayer, self).__init__(None, pass_type)

    def initialize(self, input_size):
        """
        Initializes the softmax layer.

        :param input_size: The size of the input to the layer.
        """
        self.output_size = input_size

    def forward_pass(self, inputs):
        """
        Computes softmax probabilities of the `inputs`

        :param inputs: The input data
        :return: The softmax probabilities of the inputs
        """
        return nnet.loss.softmax(inputs)


class LossLayer(NeuralNetworkLayer):
    """
    On a forward pass, this layer computes the loss and the gradient on it's inputs.
    """

    def __init__(self, pass_type="train", loss_type="softmax"):
        """
        Initialize this layer.

        :param pass_type: The type of forward pass.
        :param loss_type: The type of loss function to use.
        """
        super(LossLayer, self).__init__(None, pass_type)
        self.output_size = 1
        self.local_gradient = None
        if loss_type == "softmax":
            self.loss_function = nnet.loss.softmax

    def forward_pass(self, inputs):
        """
        Computes the loss w.r.t. the `inputs`.

        :param inputs: The input data
        :return: The loss of the forward propogation
        """
        loss, self.local_gradient = self.loss_function.gradient(inputs, self.current_batch_Y)
        return loss

    def backward_pass(self, previous_gradient=1):
        """
        Performs scalar multiplication of previous gradient with this gradient.

        :param previous_gradient: The gradient of the previous layer. If this layer is the last layer, `previous_gradient` should most definitely be equal to 1.
        :return: The gradient of the loss function w.r.t. the forward input data.
        """
        return self.local_gradient * previous_gradient


class ActivationLayer(NeuralNetworkLayer):
    """
    A activation layer.

    Uses a non-linearity to squash the output of the previous layer.
    """
    def __init__(self, pass_type="train", activation_type="relu"):
        """
        Specify the forward pass type and the type of activation function to use.

        :param pass_type: The type of forward pass.
        :param activation_type: The activation function to use.
        """
        super(ActivationLayer, self).__init__(None, pass_type)

        self.local_gradient = None

        if activation_type == "relu":
            self.activation_function = nnet.activation.relu
        elif activation_type == "leaky_relu":
            self.activation_function = nnet.activation.leaky_relu
        elif activation_type == "sigmoid":
            self.activation_function = nnet.activation.sigmoid
        elif activation_type == "tanh":
            self.activation_function = nnet.activation.tanh

    def initialize(self, input_size):
        """
        Initializes the activation layer.

        :param input_size: The size of the input to the layer.
        """
        self.output_size = input_size

    def forward_pass(self, inputs):
        """
        Performs a forward pass at this layer. Simply passes the inputs through
        an activation function and caches it's local gradient.

        :param inputs: the inputs to the layer
        :return: The output of this layer
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
