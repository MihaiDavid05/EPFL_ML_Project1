import numpy as np
from utils.algo import compute_gradient, compute_loss
from utils.data import batch_iter


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


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.
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
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            # compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma * gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def least_squares(y, tx):
    """
    Calculate the least squares solution.
    :param y:
    :param tx:
    :return Tuple<>: mse and optimal weights
    """
    optimal_weights = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    mse = compute_loss(y, tx, optimal_weights)
    return mse, optimal_weights


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression implementation.
    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    return np.linalg.solve(np.dot(tx.T, tx) + lambda_ * 2 * tx.shape[0] * np.identity(tx.shape[1]), np.dot(tx.T, y))

# TODO: add comments

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def hypothesis_linear_regression(tx, w):
    return tx @ w

def hypothesis_logistic_regression(tx, w):
    return sigmoid(hypothesis_linear_regression(tx, w))

def compute_loss_logistic_regression(y, tx, w):
    N = y.shape[0]
    h = hypothesis_logr(tx, w)

    return -1/N * ((y.T @ np.log(h)) + (1 - y.T) @ np.log(1 - h))

def compute_loss_logistic_regression_regularized(y, tx, w, lambda_):
    N = y.shape[0]

    return compute_loss_logistic_regression(y, tx, w) + lambda_/(2 * N) * w.T @ w

def compute_gradient_logistic_regression(y, tx, w):
    N = y.shape[0]
    h = hypothesis_logr(tx, w)

    return tx.T @ (h - y) / N

def compute_gradient_logistic_regression_regularized(y, tx, w, lambda_):
    N = y.shape[0]

    return compute_gradient_logistic_regression(y, tx, w) + lambda_/N * np.sum(w)

def generic_gradient_descent(y, tx, lambda_, initial_w, max_iters, gamma, comp_gradient, comp_loss):
    w = initial_w

    for n_iter in range(max_iters):

        gr = comp_gradient(y, tx, w, lambda_)
        loss = comp_loss(y, tx, w, lambda_)

        w = w - gamma * gr

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return generic_gradient_descent(y, tx, 0, initial_w, max_iters, gamma, compute_gradient_logistic_regression_regularized, compute_loss_logistic_regression_regularized)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return generic_gradient_descent(y, tx, lambda_, initial_w, max_iters, gamma, compute_gradient_logistic_regression_regularized, compute_loss_logistic_regression_regularized)
