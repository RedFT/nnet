#!/bin/bash

# Downloads and extracts the CIFAR-10 dataset
# After this script finishes executing, you can run dmgen.py [num_frames]

cifar10_link='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

wget $cifar10_link

cifar10_archive=$(basename $cifar10_link)

tar xvzf $cifar10_archive
rm $cifar10_archive -rf

`mv cifar-10-batches-py data/cifar-10`
