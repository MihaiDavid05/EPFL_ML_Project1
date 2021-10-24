from utils.algo import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Least squared gradient descent algorithm.
    :param y: Labels.
    :param tx: Features.
    :param initial_w: Initial weights vector.
    :param max_iters: Number of iterations.
    :param gamma: Gamma parameter.
    :return: The last computed weight vector and its associated loss.
    """
    # Define parameters to store w
    w = initial_w
    loss = None
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.
    :param y: Labels.
    :param tx: Features.
    :param initial_w: Initial weights vector.
    :param max_iters: Number of iterations.
    :param gamma: Gamma parameter.
    :return: The last computed weights vector and its associated loss.
    """
    # Define parameters to store w
    loss = None
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            # compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma * gradient
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def least_squares(y, tx):
    """
    Calculate the least squares solution.
    :param y: Labels.
    :param tx: Features
    :return: Optimal weights vector and its associated loss by MSE.
    """
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression implementation.
    :param y: Labels.
    :param tx: Features.
    :param lambda_: Lambda parameter.
    :return: Optimal weights vector and its associated loss by MSE.
    """
    w = np.linalg.solve(np.dot(tx.T, tx) + lambda_ * 2 * tx.shape[0] * np.identity(tx.shape[1]), np.dot(tx.T, y))
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform simple logistic regression.
    :param y: Labels.
    :param tx: Features.
    :param initial_w: Initial weights vector.
    :param max_iters: Number of iterations.
    :param gamma: Gamma parameter.
    :return: Last computed weights vector and its associated loss.
    """
    return generic_gradient_descent(y, tx, 0, initial_w, max_iters, gamma,
                                    compute_gradient_logistic_regression_regularized,
                                    compute_loss_logistic_regression_regularized)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression.
    :param y: Labels.
    :param tx: Features.
    :param lambda_: Lambda parameter.
    :param initial_w: Initials weights vector.
    :param max_iters: Number of iterations.
    :param gamma: Gamma parameter.
    :return: Last computed weights vector and its associated loss.
    """
    return generic_gradient_descent(y, tx, lambda_, initial_w, max_iters, gamma,
                                    compute_gradient_logistic_regression_regularized,
                                    compute_loss_logistic_regression_regularized)


def model(y, x, c):
    """
    Choose a specific model according to configuration parameters and get loss and weights.
    :param y: Labels.
    :param x: Input data.
    :param c: Configuration parameters.
    :return: Weights and training loss.
    """
    if c['lambda'] is not None:
        if c['model'] == 'ridge':
            w, tr_loss = ridge_regression(y, x, c['lambda'])
        else:
            w, tr_loss = reg_logistic_regression(y, x, c['lambda'], np.zeros((x.shape[1], 1)),
                                                 c['max_iters'], c['gamma'])
    else:
        w, tr_loss = logistic_regression(y, x, np.zeros((x.shape[1], 1)), c['max_iters'],
                                         c['gamma'])
    return w, tr_loss
