import numpy as np


def softmax(scores):
    scores_maxed = np.max(scores, axis=0)
    scores_fixed = scores - scores_maxed
    exponentiated = np.exp(scores_fixed)
    denominator = np.sum(exponentiated, axis=0)
    probabilities = exponentiated / denominator
    return probabilities


def softmax_gradient(S, Y):
    num_images = Y.shape[0]
    softmax_result = softmax(S)
    gradient = softmax_result.copy()
    gradient[Y, np.arange(num_images)] -= 1

    softmax_result = softmax_result[Y, np.arange(num_images)]
    softmax_result[softmax_result == 0] = np.finfo(float).min
    loss = -np.log(softmax_result)
    loss[loss == np.nan] = np.inf
    loss = np.mean(loss)

    return loss, gradient


softmax.gradient = softmax_gradient
