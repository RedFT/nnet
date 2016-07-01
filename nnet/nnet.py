import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.layers = []
        self.train_layers = None
        self.test_layers = None
        self.learning_rate = 0.001
        self.regularization_strength = 1

    def batch_info_broadcast(self, X, Y):
        for layer in self.layers:
            layer.receive_current_batch_info(X, Y)

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

    def mock_train(self, Xtr, Ytr, verbose=True):
        self.train_layers = [layer for layer in self.layers if "train" in layer.pass_type]
        self.test_layers = [layer for layer in self.layers if "test" in layer.pass_type]

        it = 0
        while 1:
            self.batch_info_broadcast(Xtr, Ytr)
            loss = self.forward_propogation(Xtr)
            self.backward_propogation()
            self.update_parameters()

            if it % 100 == 0 and verbose == True:
                acc = self.test(Xtr, Ytr)
                print("Loss: " + str(loss) + "  Accuracy: " + str(acc))

    def train(self, Xtr, Ytr, Xte, Yte, verbose=True):
        self.train_layers = [layer for layer in self.layers if "train" in layer.pass_type]
        self.test_layers = [layer for layer in self.layers if "test" in layer.pass_type]

        self.print_info()

        mask = np.random.choice(Xtr.shape[0], 512, replace=False)
        batch_x = Xtr[mask]
        batch_y = Ytr[mask]
        self.batch_info_broadcast(batch_x, batch_y)
        loss = self.forward_propogation(batch_x)
        acc = self.test(Xtr, Ytr)
        print("Loss: " + str(loss) + "  Accuracy: " + str(acc))

        it = 0
        while 1:
            mask = np.random.choice(Xtr.shape[0], 512, replace=False)
            batch_x = Xtr[mask]
            batch_y = Ytr[mask]
            self.batch_info_broadcast(batch_x, batch_y)
            loss = self.forward_propogation(batch_x)
            self.backward_propogation()
            self.update_parameters()

            if it % 100 == 0 and verbose == True:
                acc = self.test(Xtr, Ytr)
                print("Loss: " + str(loss) + "  Accuracy: " + str(acc))

            it += 1

    def test(self, Xte, Yte):
        self.batch_info_broadcast(Xte, Yte)
        probabilities = self.forward_propogation(Xte, pass_type="test")
        Yte_predicted = np.argmax(probabilities, axis=0)
        return np.mean(Yte_predicted == Yte)

    def forward_propogation(self, inputs, pass_type="train"):
        """
        Performs a forward pass through the network.

        :param passtype: The type of pass. ("train" or "test"
        :param inputs: The input to the neural network
        :return: The output (scores) of the neural network
        """
        previous_layer_output = inputs.T
        if pass_type == "train":
            for layer in self.train_layers:
                previous_layer_output = layer.forward_pass(previous_layer_output)

        elif pass_type == "test":
            for layer in self.test_layers:
                previous_layer_output = layer.forward_pass(previous_layer_output)

        return previous_layer_output

    def backward_propogation(self):
        """
        Performs a forward pass through the network.

        :param dS: Gradient of loss wrt the output (scores) of the neural network
        :return: The output (scores) of the neural network
        """
        previous_layer_gradient = 1
        for layer in reversed(self.train_layers):
            previous_layer_gradient = layer.backward_pass(previous_layer_gradient)

    def update_parameters(self):
        for layer in self.train_layers:
            layer.update_parameters(self.learning_rate, self.regularization_strength)
