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
    w = initial_w

    loss = None
    for n_iter in range(max_iters):
        gr = comp_gradient(y, tx, w, lambda_)
        loss = comp_loss(y, tx, w, lambda_)

        w = w - gamma * gr

    return w, loss


def error(preds, labels):
    """
    Calculate error.
    :param preds:
    :param labels:
    :return:
    """
    return np.sum(np.absolute(preds - labels))/labels.shape[0]


def accuracy(preds, labels):
    """
    Calculate accuracy.
    :param preds:
    :param labels:
    :return:
    """
    return 1 - error(preds, labels)


def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix.
    :param weights:
    :param data:
    :return:
    """
    y_pred = np.dot(data, weights)
    # y_pred[np.where(y_pred <= 0)] = -1
    # y_pred[np.where(y_pred > 0)] = 1
    # TODO: vezi aici
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def do_cross_validation(folds, model, config):
    """
    Perform cross validation
    :param folds:
    :param model:
    :param config:
    :return:
    """
    final_val_acc = 0
    folds = np.array(folds)
    for i, fold in enumerate(folds):
        # Get validation data and labels for the validation fold
        val_feats, val_labels = (fold[:, :-1], fold[:, -1].reshape((-1, 1)))
        # Set training data and labels as the rest of the folds
        train_split = np.vstack([fold for j, fold in enumerate(folds) if j != i])
        tr_feats, tr_labels = (train_split[:, :-1], train_split[:, -1].reshape((-1, 1)))
        # Find weights
        weights, tr_loss = model(tr_labels, tr_feats, np.zeros((val_feats.shape[1], 1)), config['max_iters'],
                                 config['gamma'])
        # Make predictions for both training and validation
        tr_preds = predict_labels(weights, tr_feats)
        val_preds = predict_labels(weights, val_feats)
        # Get accuracy for training and validation
        tr_acc = accuracy(tr_preds, tr_labels)
        val_acc = accuracy(val_preds, val_labels)
        # Sum validation accuracy
        final_val_acc += val_acc
        print("For fold {}, training accuracy is {:.2f} % and validation accuracy is {:.2f} %".format(i + 1,
                                                                                                      tr_acc * 100,
                                                                                                      val_acc * 100))
    # Compute final validation accuracy
    final_val_acc = final_val_acc / len(folds)

    return final_val_acc
