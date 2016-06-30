import numpy as np
import nnet
import test


Xtr, Ytr, Xte, Yte, label_names = test.get_cifar10_dataset()

# PRE-PROCESSING
Xtr = test.normalize(Xtr)
Xte = test.normalize(Xte)

mean = np.mean(np.concatenate([Xtr, Xte]), axis=0)
Xtr = Xtr - mean
Xte = Xte - mean

Xtr = test.append_one(Xtr)
Xte = test.append_one(Xte)


# Neural Net
nn = nnet.NeuralNetwork(Xtr.shape[1])

nn.add_layer(nnet.FullyConnectedLayer(initialization_type="xavier_relu", output_size=100, pass_type="test|train"))
nn.add_layer(nnet.ActivationLayer("relu", pass_type="test|train"))
nn.add_layer(nnet.FullyConnectedLayer(initialization_type="xavier_relu", output_size=10, pass_type="test|train"))
nn.add_layer(nnet.SoftmaxLayer(pass_type="test"))
nn.add_layer(nnet.LossLayer("softmax", pass_type="train"))

nn.train(Xtr, Ytr, Xte, Yte)

