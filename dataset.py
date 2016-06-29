import numpy as np


def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def getCifarDataset():
    data = []
    for i in range(1, 6):
        data.append(unpickle("data/cifar-10/data_batch_" + str(i)))

    images = np.concatenate([item["data"] for item in data])
    labels = np.concatenate([item["labels"] for item in data])

    testing_data = unpickle("data/cifar-10/test_batch")
    label_names = unpickle("data/cifar-10/batches.meta")

    return images, labels, testing_data["data"], np.array(testing_data["labels"]), label_names["label_names"]


if __name__ == "__main__":
    dataset = getCifarDataset()
    for item in dataset:
        print(item.shape)
