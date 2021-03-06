[![Build Status](https://travis-ci.org/RedFT/nnet.svg?branch=master)](https://travis-ci.org/RedFT/nnet)


## Synopsis

This is a simple N-layer Neural Network library.
[This project uses Python 3]


## Motivation

I am following Stanford's CS231n course (Convolutional Neural Networks for Visual Recognition). I decided I would build a small Neural Network library as I progress through this course.


## Code Example

Below is an example of a two layer neural network. You can find similar code in `test/run_tests.py`.
```python
nn = nnet.NeuralNetwork(input_size)

nn.add_layer(nnet.FullyConnectedLayer(
    initialization_type="xavier_relu", 
    output_size=100, 
    pass_type="test|train"))
nn.add_layer(nnet.ActivationLayer(
    "relu", 
    pass_type="test|train"))
nn.add_layer(nnet.FullyConnectedLayer(
    initialization_type="xavier_relu", 
    output_size=10, 
    pass_type="test|train"))
nn.add_layer(nnet.SoftmaxLayer(
    pass_type="test"))
nn.add_layer(nnet.LossLayer(
    "softmax", 
    pass_type="train"))

nn.train(training_data, training_labels, 
    testing_data, testing_labels)
```

## Documentation

Documentation is minimal. 

To generate html documentation, you must first install [sphinx](https://http://www.sphinx-doc.org/).
```bash
$ pip3 install sphinx
```

Change into the `docs` directory. You can then generate html documentation by running:
```bash
$ make html
```

To view the html documentation in your web browser (i.e. `firefox`), you can run:
```bash
$ firefox build/html/index.html
```

## Install Dependencies

From the project's root directory run:
```bash
$ pip3 install -r requirements.txt
```

## Test

Change into the `test` directory.
To run the test script, you need to make sure you have the Cifar10 dataset in the `data` directory. Just run:
```bash
$ chmod +x get_cifar10_dataset.sh
$ ./get_cifar10_dataset.sh
```

Then from the project's root directory:
```bash
$ nosetests -v
```

## License

[![GNU GPL v3.0](http://www.gnu.org/graphics/gplv3-127x51.png)](http://www.gnu.org/licenses/gpl.html)

This code in this project is released under the GPL License.
