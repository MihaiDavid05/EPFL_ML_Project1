import csv
import numpy as np
from random import randrange


def load_csv_data(data_path, sub_sample=False):
    """
    Loads data.
    :param data_path:
    :param sub_sample:
    :return: y (class labels), tX (features), ids (event ids) and features names
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    feats_name = np.genfromtxt(data_path, delimiter=",", max_rows=1, autostrip=True, dtype='str')[2:]
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    # TODO: Check here it should be -1 not 0
    yb[np.where(y == 'b')] = 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids, feats_name


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    :param ids: event ids associated with each prediction
    :param y_pred: predicted class labels
    :param name: string name of .csv output file to be created
    :return:
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            # TODO: Check here at Prediction
            writer.writerow({'Id': int(r1), 'Prediction': int(r2) if int(r2) == 1 else -1})


def normalize(x):
    """
    Standardize the original data set.
    :param x:
    :return:
    """
    # Transform to zero mean
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    # Transform to unit standard deviation
    std_x = np.std(x, axis=0)
    x = x / (std_x + 0.0000001)
    return x, mean_x, std_x


def build_model_data(feats):
    """
    Get necessary format for feats and labels.
    :param feats:
    :return:
    """
    num_samples = feats.shape[0]
    # Build matrix 'tilda' x
    tx = np.c_[np.ones(num_samples), feats]
    return tx


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


def split_data(x, y, ratio=0.8, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to validation
    :param x:
    :param y:
    :param ratio:
    :param seed:
    :return:
    """
    # set seed
    np.random.seed(seed)
    # Randomly select indexes for training and validation
    num_row = y.shape[0]
    indices = np.random.RandomState(seed=seed).permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[:index_split]
    index_val = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_val = x[index_val]
    y_tr = y[index_tr]
    y_val = y[index_val]
    return x_tr, x_val, y_tr, y_val


def cross_validation_split(dataset, folds=5):
    """
    Create folds for cross validation.
    :param dataset:
    :param folds:
    :return:
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: input data
    :param degree: polynomial degree
    :return: matrix formed by applying the polynomial basis to the input data. All features to power 1, then to power 2 and so on.
    """
    nr_feats = x.shape[1]
    final_matrix = np.zeros((x.shape[0], nr_feats + (degree - 1) * nr_feats + 1))
    final_matrix[:, 0] = np.ones((final_matrix.shape[0],))
    for j in range(1, degree+1):
        for k in range(nr_feats):
            final_matrix[:, 1 + k + ((j - 1) * x.shape[1])] = np.power(x[:, k], j)
    return final_matrix


def replace_values(config, feats):
    """
    Replace values of -999 with zeros or feature-wise mean.
    :param config:
    :param feats:
    :return:
    """
    if config["replace_with"] == 'mean':
        # TODO: Implement this
        pass
    elif config["replace_with"] == 'zero':
        feats = np.where(feats == float(-999), float(0), feats)
    return feats
