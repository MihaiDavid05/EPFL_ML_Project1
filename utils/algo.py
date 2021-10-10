import numpy as np


def compute_gradient(y, tx, w):
    """
    Compute the gradient.
    :param y:
    :param tx:
    :param w:
    :return:
    """
    e = y - np.dot(tx, w)
    return (-1 / tx.shape[0]) * np.dot(tx.T, e)


def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
    :param y:
    :param tx:
    :param w:
    :return:
    """
    e = y - np.dot(tx, w)
    return (-1 / tx.shape[0]) * np.dot(tx.T, e)


def compute_loss(y, tx, w):
    """
    Calculate the loss by MSE.
    :param y:
    :param tx:
    :param w:
    :return:
    """
    e = y - np.dot(tx, w)
    return np.dot(e.T, e) / (2*tx.shape[0])


def compute_loss_mae(y, tx, w):
    """
    Calculate the loss by MAE.
    :param y:
    :param tx:
    :param w:
    :return:
    """
    e = y - np.dot(tx, w)
    return np.sum(np.absolute(e)) / (tx.shape[0])


def sigmoid(v):
    # print("Aici")
    return 1 / (1 + np.exp(-v))


def hypothesis_linear_regression(tx, w):
    return tx @ w


def hypothesis_logistic_regression(tx, w):
    return sigmoid(hypothesis_linear_regression(tx, w))


def compute_loss_logistic_regression(y, tx, w):
    N = y.shape[0]
    h = hypothesis_logistic_regression(tx, w)

    return -1 / N * ((y.T @ np.log(h)) + (1 - y.T) @ np.log(1 - h))


def compute_loss_logistic_regression_regularized(y, tx, w, lambda_):
    N = y.shape[0]

    return compute_loss_logistic_regression(y, tx, w) + lambda_ / (2 * N) * w.T @ w


def compute_gradient_logistic_regression(y, tx, w):
    N = y.shape[0]
    h = hypothesis_logistic_regression(tx, w)
    return tx.T @ (h - y) / N


def compute_gradient_logistic_regression_regularized(y, tx, w, lambda_):
    N = y.shape[0]

    return compute_gradient_logistic_regression(y, tx, w) + lambda_ / N * np.sum(w)


def generic_gradient_descent(y, tx, lambda_, initial_w, max_iters, gamma, comp_gradient, comp_loss):
    w = initial_w

    loss = None
    for n_iter in range(max_iters):
        gr = comp_gradient(y, tx, w, lambda_)
        loss = comp_loss(y, tx, w, lambda_)

        w = w - gamma * gr

    return w, loss


def error(preds, labels):
    return np.sum(np.absolute(preds - labels))/labels.shape[0]


def accuracy(preds, labels):
    return 1 - error(preds, labels)
