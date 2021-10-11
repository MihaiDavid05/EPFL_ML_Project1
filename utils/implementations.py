from utils.algo import *
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
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient
        # store loss
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, w


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
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            # compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma * gradient
            # store loss
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, w


def least_squares(y, tx):
    """
    Calculate the least squares solution.
    :param y:
    :param tx:
    :return Tuple<>: mse and optimal weights
    """
    optimal_weights = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    mse_loss = compute_loss(y, tx, optimal_weights)
    return mse_loss, optimal_weights


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression implementation.
    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    return np.linalg.solve(np.dot(tx.T, tx) + lambda_ * 2 * tx.shape[0] * np.identity(tx.shape[1]), np.dot(tx.T, y))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform simple logistic regression.
    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    return generic_gradient_descent(y, tx, 0, initial_w, max_iters, gamma,
                                    compute_gradient_logistic_regression_regularized,
                                    compute_loss_logistic_regression_regularized)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression
    :param y:
    :param tx:
    :param lambda_:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    return generic_gradient_descent(y, tx, lambda_, initial_w, max_iters, gamma,
                                    compute_gradient_logistic_regression_regularized,
                                    compute_loss_logistic_regression_regularized)
