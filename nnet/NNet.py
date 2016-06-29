import nnet


class NeuralNetwork(object):
    def __init__(self, input_size=None):
        super(NeuralNetwork, self).__init__()
        self.input_size=input_size
        self.layers = []

    def add_layer(self, layer, input_size=None):
        current_layer_input_size = None
        if len(self.layers) == 0:
            if self.input_size is not None:
                current_layer_input_size = self.input_size
            else:
                current_layer_input_size = input_size

        else:
            current_layer_input_size = self.layers[-1].output_size

        layer.initialize(current_layer_input_size)
        self.layers.append(layer)


if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.add_layer(nnet.FullyConnectedLayer())

