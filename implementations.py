import numpy as np


def compute_gradient(y, tx, w):
    """
    Compute the gradient.
    :param y:
    :param tx:
    :param w:
    :return:
    """
    # compute gradient and error vector
    e = y - np.dot(tx, w)
    return (-1 / tx.shape[0]) * np.dot(tx.T, e)


def compute_loss(y, tx, w):
    """
    Calculate the loss.
    :param y:
    :param tx:
    :param w:
    :return:
    """
    # compute loss by MSE
    e = y - np.dot(tx, w)
    return np.dot(e.T, e) / (2*tx.shape[0])


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Least squared gradient descent algorithm.
    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws