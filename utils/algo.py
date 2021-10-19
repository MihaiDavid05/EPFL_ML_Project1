import numpy as np


def compute_gradient(y, tx, w):
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
    return 1 / (1 + np.exp(-v))


def hypothesis_linear_regression(tx, w):
    return tx @ w


def hypothesis_logistic_regression(tx, w):
    return sigmoid(hypothesis_linear_regression(tx, w))


def compute_loss_logistic_regression(y, tx, w):
    n = y.shape[0]
    h = hypothesis_logistic_regression(tx, w)

    return -1 / n * ((y.T @ np.log(h)) + (1 - y.T) @ np.log(1 - h))


def compute_loss_logistic_regression_regularized(y, tx, w, lambda_):
    n = y.shape[0]

    return compute_loss_logistic_regression(y, tx, w) + lambda_ / (2 * n) * w.T @ w


def compute_gradient_logistic_regression(y, tx, w):
    n = y.shape[0]
    h = hypothesis_logistic_regression(tx, w)
    return tx.T @ (h - y) / n


def compute_gradient_logistic_regression_regularized(y, tx, w, lambda_):
    n = y.shape[0]

    return compute_gradient_logistic_regression(y, tx, w) + lambda_ / n * np.sum(w)


def generic_gradient_descent(y, tx, lambda_, initial_w, max_iters, gamma, comp_gradient, comp_loss):
    """
    Generic function for computing weights and loss.
    :param y:
    :param tx:
    :param lambda_:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :param comp_gradient:
    :param comp_loss:
    :return:
    """
    w = initial_w
    loss = None
    for n_iter in range(max_iters):
        gr = comp_gradient(y, tx, w, lambda_)
        loss = comp_loss(y, tx, w, lambda_)
        w = w - gamma * gr
    return w, loss


def get_precision_recall_accuracy(preds, labels):
    """
    Compute precision, recall and accuracy.
    :param preds:
    :param labels:
    :return:
    """
    preds = np.ravel(preds)
    labels = np.ravel(labels)
    tp = len(np.where(np.logical_and(preds == 1, labels == 1))[0])
    fp = len(np.where(np.logical_and(preds == 1, labels == 0))[0])
    fn = len(np.where(np.logical_and(preds == 0, labels == 1))[0])
    tn = len(np.where(np.logical_and(preds == 0, labels == 0))[0])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return precision, recall, accuracy


def get_f1(preds, labels):
    """
    Compute f1 score.
    :param preds:
    :param labels:
    :return:
    """
    precision, recall, _ = get_precision_recall_accuracy(preds, labels)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def predict_labels(weights, data, threshold=0.5):
    """
    Generates class predictions given weights, and a test data matrix.
    :param weights:
    :param data:
    :param threshold
    :return:
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = 0
    y_pred[np.where(y_pred > threshold)] = 1

    return y_pred


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables 'y' and 'tx'.
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    :param y: output desired values
    :param tx: input data
    :param batch_size: size of batch
    :param num_batches: number of batches
    :param shuffle: shuffle the data or not
    :return:
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
