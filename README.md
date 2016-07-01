## Synopsis

This is a simple N-layer Neural Network library.

## Code Example

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
$ pip install sphinx
```

You can generate html documentation by running:
```bash
$ make html
```

To view the html documentation in your web browser (i.e. `firefox`, you can run:
```bash
$ firefox build/index.html
```

## Motivation

I am following Stanford's CS231n course (Convolutional Neural Networks for Visual Recognition). I decided I would build a small Neural Network library as I progress through this course.

## Install Dependencies

From the project's root directory run:
```bash
$ pip install -r requirements.txt
```

## Test

To run the test script, you need to make sure you have the Cifar10 dataset in the `data` directory. Just run:
```bash
$ chmod +x get_cifar10_dataset.sh
$ ./get_cifar10_dataset.sh
```

Then run the test.py script:
```bash
$ python test.py
```

## License

GPL
