import sys
sys.path.insert(0, "..")
import nnet
import test
import numpy as np


def two_layer_test():
    Xtr, Ytr, Xte, Yte, label_names = test.get_cifar10_dataset()

    # Reshape each data point to be a 1-dimensional array, for a plain neural network.
    Xtr = Xtr.reshape(50000, 32 * 32 * 3)
    Xte = Xte.reshape(10000, 32 * 32 * 3)

    # PRE-PROCESSING
    Xtr = test.normalize(Xtr)
    Xte = test.normalize(Xte)

    mean = np.mean(np.concatenate([Xtr, Xte]), axis=0)
    Xtr = Xtr - mean
    Xte = Xte - mean

    Xtr = test.append_zeros(Xtr)
    Xte = test.append_zeros(Xte)

    # Neural Net
    nn = nnet.NeuralNetwork(Xtr.shape[1])
    nn.batch_size = 512

    nn.set_training_set(Xtr.T, Ytr)
    nn.set_testing_set(Xte.T, Yte)

    nn.add_layer(nnet.FullyConnectedLayer(pass_type="test|train", output_size=100, initialization_type='xavier'))
    nn.add_layer(nnet.BatchNormalizationLayer(pass_type="test|train"))
    nn.add_layer(nnet.ActivationLayer(pass_type="test|train", activation_type="leaky_relu"))
    nn.add_layer(nnet.FullyConnectedLayer(pass_type="test|train", output_size=10, initialization_type='xavier'))
    nn.add_layer(nnet.SoftmaxLayer(pass_type="test"))
    nn.add_layer(nnet.LossLayer(pass_type="train"))

    # Print out each layer's information in order, then train.
    nn.print_info()
    final_loss, final_accuracy = nn.train(iterations=500)
    assert(final_loss < 1)
    assert(final_accuracy > 0.30)


def three_layer_test():
    Xtr, Ytr, Xte, Yte, label_names = test.get_cifar10_dataset()

    # Reshape each data point to be a 1-dimensional array, for a plain neural network.
    Xtr = Xtr.reshape(50000, 32 * 32 * 3)
    Xte = Xte.reshape(10000, 32 * 32 * 3)

    # PRE-PROCESSING
    Xtr = test.normalize(Xtr)
    Xte = test.normalize(Xte)

    mean = np.mean(np.concatenate([Xtr, Xte]), axis=0)
    Xtr = Xtr - mean
    Xte = Xte - mean

    Xtr = test.append_zeros(Xtr)
    Xte = test.append_zeros(Xte)

    # Neural Net
    nn = nnet.NeuralNetwork(Xtr.shape[1])
    nn.batch_size = 512

    nn.set_training_set(Xtr.T, Ytr)
    nn.set_testing_set(Xte.T, Yte)

    nn.add_layer(nnet.FullyConnectedLayer(pass_type="test|train", output_size=100, initialization_type='xavier'))
    nn.add_layer(nnet.BatchNormalizationLayer(pass_type="test|train"))
    nn.add_layer(nnet.ActivationLayer(pass_type="test|train", activation_type="leaky_relu"))

    nn.add_layer(nnet.FullyConnectedLayer(pass_type="test|train", output_size=50, initialization_type='xavier'))
    nn.add_layer(nnet.BatchNormalizationLayer(pass_type="test|train"))
    nn.add_layer(nnet.ActivationLayer(pass_type="test|train", activation_type="leaky_relu"))

    nn.add_layer(nnet.FullyConnectedLayer(pass_type="test|train", output_size=10, initialization_type='xavier'))
    nn.add_layer(nnet.SoftmaxLayer(pass_type="test"))
    nn.add_layer(nnet.LossLayer(pass_type="train"))

    # Print out each layer's information in order, then train.
    nn.print_info()
    final_loss, final_accuracy = nn.train(iterations=500)
    assert(final_loss < 1)
    assert(final_accuracy > 0.30)
