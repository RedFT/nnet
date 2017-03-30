import math

import numpy as np
import sys

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


class ConvolutionalLayer(NeuralNetworkLayer):
    def __init__(self, num_filters, filter_size, stride, zero_padding, initialization_type="xavier", output_size=None, pass_type="train|test"):
        super(ConvolutionalLayer, self).__init__(output_size, pass_type)
        self.initialization_type = initialization_type
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.zero_padding = zero_padding
        self.output_size = None
        self.filters = None

    def initialize(self, input_size):
        self.input_size = input_size
        w, h, d = input_size
        self.output_size = (
            ((h - self.filter_size + 2 * self.zero_padding) // self.stride) + 1,
            ((w - self.filter_size + 2 * self.zero_padding) // self.stride) + 1,
            self.num_filters)
        self.filters = np.random.rand(self.num_filters, self.filter_size, self.filter_size, d)
        self.dFilters = np.zeros(shape=self.filters.shape)
        if self.initialization_type == "xavier":
            w_bound = math.sqrt(w * h * d)
            self.filters /= w_bound
        elif self.initialization_type == "xavier_relu":
            w_bound = math.sqrt(w * h * d / 2)
            self.filters /= w_bound
        elif self.initialization_type == "gaussian":
            self.filters /= 0.01
        else:
            print("Error: initialization Type '" + self.initialization_type + "' not recognized")
            exit(1)

    def forward_pass(self, inputs):
        self.forward_input = inputs
        num_examples = inputs.shape[0]

        w, h, d = self.input_size
        output = np.zeros([num_examples] + list(self.output_size))
        for ex_idx in range(num_examples):
            for filter_idx in range(len(self.filters)):
                for y in range(0, h, self.stride):
                    if y + self.filter_size > h:
                        break

                    for x in range(0, w, self.stride):
                        if x + self.filter_size > w:
                            break

                        dot = np.sum(inputs[ex_idx][y:y+self.stride + 1, x:x+self.stride + 1,:] * self.filters[filter_idx])
                        out_y = y // self.stride
                        out_x = x // self.stride
                        output[ex_idx, out_y, out_x] = dot
        return output

    def backward_pass(self, previous_gradient):
        w, h, depth = previous_gradient.shape
        for d in range(0, depth):
            dFilter = np.zeros(shape=(self.filter_size, self.filter_size))
            for y in range(0, h):
                for x in range(0, w):
                    orig_x = x * self.stride
                    orig_y = y * self.stride

                    dAdd = np.ones(shape=(filter, filter, d)) * previous_gradient[y, x]

                    local_gradient = self.forward_input[
                            orig_y:orig_y + self.stride + 1,
                            orig_x:orig_x + self.stride + 1,:].T

                    dFilter += dAdd * local_gradient
            self.dFilters[d] = dFilter / previous_gradient.size

    def update_parameters(self, learning_rate, regularization_strength):
        for i in range(len(self.dFilters)):
            self.filters[i] += -learning_rate * self.dFilters[i]




class PoolingLayer(NeuralNetworkLayer):
    """
    Not implemented for now
    """


class FullyConnectedLayer(NeuralNetworkLayer):
    """
    During a forward pass, this layer performs a dot product operation on its input with its weights.

    During a backward pass, this layer performs a dot product operation on its forward input transposed
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
        self.input_size = input_size

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

        :param inputs: The vectors to multiply the weight with. N x D
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
    on its inputs.
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
    On a forward pass, this layer computes the loss and the gradient on its inputs.
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
        an activation function and caches its local gradient.

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


class BatchNormalizationLayer(NeuralNetworkLayer):
    """
    Implementation of a batch normalization layer.
    """
    def __init__(self, pass_type="test|train"):
        super(BatchNormalizationLayer, self).__init__(pass_type)
        self.gamma = None
        self.beta = None

        self.input_mean = None
        self.input_mean_subtracted = None

        self.input_mean_subtracted_squared = None
        self.input_variance = None
        self.input_standard_deviation = None

        self.input_standard_deviation_inverse = None
        self.input_normalized = None

        self.scaled = None
        self.shifted = None

    def initialize(self, input_size):
        """
        Initializes the batch normalization layer.

        :param input_size: The size of the input to the layer.
        """
        self.output_size = input_size
        self.gamma = np.ones(shape=(input_size, 1))
        self.beta = 0

    def forward_pass(self, inputs):
        """
        Normalizes the inputs. Then scales and shifts the normalized data with learned parameters, gamma and beta

        :param inputs: The input to the layer
        :return: The normalized inputs, shifted and scaled by learned parameters, gamma and beta
        """
        # subtract the mean
        self.input_mean = inputs.mean()
        self.input_mean_subtracted = inputs - self.input_mean

        # calculate standard deviation
        self.input_mean_subtracted_squared = self.input_mean_subtracted * self.input_mean_subtracted
        self.input_variance = self.input_mean_subtracted_squared.mean()
        self.input_standard_deviation = math.sqrt(self.input_variance - sys.float_info.epsilon)

        # divide mean subtracted input by the standard deviation of inputs
        self.input_standard_deviation_inverse = 1 / self.input_standard_deviation
        self.input_normalized = self.input_mean_subtracted * self.input_standard_deviation_inverse

        # shift and scale to be "better" for the next layer
        self.scaled = self.input_normalized * self.gamma
        self.shifted = self.scaled + self.beta
        return self.shifted

    def backward_pass(self, previous_gradient):
        num_examples = previous_gradient.shape[1]

        self.dBeta = previous_gradient.sum(axis=0)
        self.dGamma = np.sum(previous_gradient * self.input_normalized, axis=0)

        dNormalized = self.gamma * previous_gradient
        dMeanSubtracted = dNormalized * self.input_standard_deviation_inverse
        dStandardDeviationInverse = dNormalized.sum(axis=0) * self.input_mean_subtracted

        dStandardDeviation = dStandardDeviationInverse * -1 / self.input_standard_deviation
        dVariance = 0.5 * 1 / math.sqrt(self.input_variance + sys.float_info.epsilon) * dStandardDeviation
        dSquared = np.ones(shape=self.input_mean_subtracted.shape) * dVariance / num_examples

        dMean1 = 2 * dMeanSubtracted * dSquared

        dForkSub = dMean1 + dMeanSubtracted
        dAverage = -1 * dForkSub

        dInput = dForkSub

        dInput2 = dAverage * np.ones(self.input_mean.shape) / num_examples
        return dInput + dInput2

    def update_parameters(self, learning_rate, regularization_strength):
        self.gamma += -learning_rate * np.mean(self.dGamma, axis=0)
        self.beta += -learning_rate * self.dBeta
