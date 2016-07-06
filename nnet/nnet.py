"""
An N-Layer Neural Network
"""
import numpy as np


class NeuralNetwork(object):
    """
    An N-Layer Neural Network
    """
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.layers = []
        self.train_layers = None
        self.test_layers = None

        self.training_data = None
        self.testing_data = None
        self.training_labels = None
        self.testing_labels = None

        # Hyperparameters
        self.learning_rate = 0.001
        self.regularization_strength = 1
        self.batch_size = 256

    def set_training_set(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    def set_testing_set(self, testing_data, testing_labels):
        self.testing_data = testing_data
        self.testing_labels = testing_labels

    def batch_info_broadcast(self, data_batch, label_batch):
        """
        Broadcasts the input data and labels to all layers.

        :param data_batch: input data
        :param label_batch: input labels
        :return: None
        """
        for layer in self.layers:
            layer.receive_current_batch_info(data_batch, label_batch)

    def add_layer(self, layer):
        """
        Initializes the layer passed in and adds it to the neural network.

        :param layer: The layer object to be added to the neural network.
        :return: None
        """
        if len(self.layers) == 0:
            current_layer_input_size = self.input_size
        else:
            current_layer_input_size = self.layers[-1].output_size

        layer.initialize(input_size=current_layer_input_size)
        self.layers.append(layer)

    def print_info(self):
        for layer in self.layers:
            print(str(layer) + " size: " + str(layer.output_size))

    def train(self, iterations=None, verbose=True):
        self.train_layers = [layer for layer in self.layers if "train" in layer.pass_type]
        self.test_layers = [layer for layer in self.layers if "test" in layer.pass_type]

        mask = np.random.choice(self.training_data.shape[0], self.batch_size, replace=False)
        batch_x = self.training_data[:,mask]
        batch_y = self.training_labels[mask]

        self.batch_info_broadcast(batch_x, batch_y)
        loss = self.forward_propogation(batch_x)
        acc = self.test()
        print("Loss: " + str(loss) + "  Accuracy: " + str(acc))

        num_iters = iterations
        if iterations is None:
            num_iters = np.inf

        it = 1
        while it < num_iters:
            mask = np.random.choice(self.training_data.shape[0], 512, replace=False)
            batch_x = self.training_data[:,mask]
            batch_y = self.training_labels[mask]
            self.batch_info_broadcast(batch_x, batch_y)
            loss = self.forward_propogation(batch_x)
            self.backward_propogation()
            self.update_parameters()

            if it % 100 == 0 and verbose == True:
                acc = self.test()
                print("Loss: " + str(loss) + "  Accuracy: " + str(acc))

            it += 1

    def test(self):
        self.batch_info_broadcast(self.testing_data, self.testing_labels)
        probabilities = self.forward_propogation(self.testing_data, pass_type="test")
        testing_labels_predicted = np.argmax(probabilities, axis=0)
        return np.mean(testing_labels_predicted == self.testing_labels)

    def forward_propogation(self, inputs, pass_type="train"):
        """
        Performs a forward pass through the network.

        :param pass_type: The type of pass. ("train" or "test"
        :param inputs: The input to the neural network
        :return: The output (scores) of the neural network
        """
        previous_layer_output = inputs
        if pass_type == "train":
            for layer in self.train_layers:
                previous_layer_output = layer.forward_pass(previous_layer_output)

        elif pass_type == "test":
            for layer in self.test_layers:
                previous_layer_output = layer.forward_pass(previous_layer_output)

        return previous_layer_output

    def backward_propogation(self):
        """
        Performs a backward pass through the network.

        :return: The output (scores) of the neural network
        """
        previous_layer_gradient = 1
        for layer in reversed(self.train_layers):
            previous_layer_gradient = layer.backward_pass(previous_layer_gradient)

    def update_parameters(self):
        for layer in self.train_layers:
            layer.update_parameters(self.learning_rate, self.regularization_strength)
