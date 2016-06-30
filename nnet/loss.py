import numpy as np


def softmax(self, scores):
    numerator = np.exp(scores)
    denominator = np.sum(np.exp(scores), axis=0)
    probabilities = numerator / denominator
    return probabilities


def softmax_gradient(self, S, Y):  # , reg, parameters):
    num_images = Y.shape[0]
    S_maxes = np.max(S, axis=0)
    S_fixed = S - S_maxes
    softmax_result = self.softmax_matrix(S_fixed)
    gradient = softmax_result.copy()
    gradient[Y, np.arange(num_images)] -= 1

    softmax_result = softmax_result[Y, np.arange(num_images)]
    softmax_result[softmax_result == 0] = np.finfo(float).min
    loss = -np.log(softmax_result)
    loss[loss == np.nan] = np.inf
    loss = np.mean(loss)

    return loss, gradient


softmax.gradient = softmax_gradient
