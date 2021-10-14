import numpy as np


def load_data(file_path, train=True):
    """
    Load data and convert it to the metrics system.
    :param file_path:
    :param train:
    :return:
    """
    feats_name = np.genfromtxt(file_path, delimiter=",", max_rows=1, autostrip=True, dtype='str')[2:]
    if train:
        labels = np.genfromtxt(file_path, delimiter=",", usecols=1, skip_header=1, dtype='str')
    else:
        labels = None
    index = np.genfromtxt(file_path, delimiter=",", usecols=0, skip_header=1, dtype='int32')
    samples_feats = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype='float32')[:, 2:]
    return feats_name, samples_feats, index, labels


def standardize(x):
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
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(feats, labels):
    """
    Form (y,tx) to get regression data in matrix form.
    :param feats:
    :param labels:
    :return:
    """
    # Decode labels into 0 and 1
    y = list(map(lambda l: 0 if l == 'b' else 1, labels))
    num_samples = feats.shape[0]
    # Build matrix 'tilda' x
    tx = np.c_[np.ones(num_samples), feats]
    return y, tx


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


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred
