import numpy as np


def unpickle(file):
    # This function is from : https://www.cs.toronto.edu/~kriz/cifar.html
    # Modified to work with Python 3
    import pickle
    fo = open(file, 'rb')
    data_dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return data_dict


def get_cifar10_dataset():
    # load data
    data = [unpickle("test/data/cifar-10-batches-py/data_batch_" + str(i)) for i in range(1, 6)]
    testing_data = unpickle("test/data/cifar-10-batches-py/test_batch")
    label_names = unpickle("test/data/cifar-10-batches-py/batches.meta")

    # combine all training data
    training_images = np.concatenate([item["data"] for item in data])
    training_labels = np.concatenate([item["labels"] for item in data])

    # rename for readability
    testing_images = testing_data["data"]
    testing_labels = np.array(testing_data["labels"])
    label_strings = label_names["label_names"]

    # reshape into images
    training_images = training_images.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    testing_images = testing_images.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    return training_images, training_labels, testing_images, testing_labels, label_strings


def normalize(X):
    return X.astype(np.float32) / 255


def append_zeros(X):
    ones_1 = np.array([np.zeros([X.shape[0]])])
    return np.append(X, ones_1.T, axis=1)
