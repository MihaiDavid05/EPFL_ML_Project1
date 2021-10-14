from data import load_csv_data, standardize, predict_labels
from implementations import *


def accuracy(labels, labels_pred):
    return np.sum(labels == labels_pred) / labels.shape[0]


def train():
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    tX = standardize(tX)[0]

    # Define the parameters of the algorithm.
    max_iters = 50
    gamma = 0.7

    lambda_ = 0.01

    # Initialization
    w_initial = np.zeros(tX.shape[1])

    ls_GD_ws, ls_GD_loss = least_squares_GD(y, tX, w_initial, max_iters, gamma)
    ls_SGD_ws, ls_SGD_loss = least_squares_SGD(y, tX, w_initial, max_iters, gamma)
    ls_ws, ls_loss = least_squares(y, tX)
    rr_ws, rr_loss = ridge_regression(y, tX, lambda_[0])
    lr_ws, lr_loss = logistic_regression(y, tX, w_initial, max_iters, gamma)

    print("LS_GD: loss={l}".format(l=ls_GD_loss))
    print("LS_SGD: loss={l}".format(l=ls_SGD_loss))
    print("LS_: loss={l}".format(l=ls_loss))
    print("RR: loss={l}".format(l=rr_loss))
    print("LR: loss={l}".format(l=lr_loss))

    y_pred_ls_GD = predict_labels(ls_GD_ws, tX)
    y_pred_ls_SGD = predict_labels(ls_SGD_ws, tX)
    y_pred_ls = predict_labels(ls_ws, tX)
    y_pred_rr = predict_labels(rr_ws, tX)
    y_pred_lr_SGD = predict_labels(lr_ws, tX)

    print(accuracy(y, y_pred_ls_GD))
    print(accuracy(y, y_pred_ls_SGD))
    print(accuracy(y, y_pred_ls))
    print(accuracy(y, y_pred_rr))
    print(accuracy(y, y_pred_lr_SGD))
