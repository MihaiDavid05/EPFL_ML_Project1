import csv
import numpy as np
import numpy.ma as ma
from random import randrange
from utils.vizualization import plot_hist_panel


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
    # TODO: Check here it should be -1, not 0
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


def standardize(x, tr_mean=None, tr_std=None):
    """
    Standardize the original data set.
    :param x:
    :param tr_mean:
    :param tr_std:
    :return:
    """
    # Transform to zero mean
    if tr_mean is None:
        mean_x = np.mean(x, axis=0)
        x = x - mean_x
    else:
        x = x - tr_mean
        mean_x = None
    # Transform to unit standard deviation
    if tr_std is None:
        std_x = np.std(x, axis=0)
        x = x / (std_x + 0.0000001)
    else:
        x = x / (tr_std + 0.0000001)
        std_x = None

    return x, mean_x, std_x


def build_model_data(feats):
    """
    Get necessary format for feats.
    :param feats:
    :return:
    """
    # Build matrix 'tilda' x
    tx = np.c_[np.ones(feats.shape[0]), feats]
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


def build_poly(x, degree, multiply_each=False):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: input data
    :param degree: polynomial degree
    :param multiply_each: Form new columns by muliplying xi * xj for i,j=0,...,nr_feats
    :return: matrix formed by applying the polynomial basis to the input data. All features to power 1, then to power 2
    and so on.
    """
    nr_feats = x.shape[1]
    final_matrix = np.zeros((x.shape[0], nr_feats + (degree - 1) * nr_feats + 1))
    final_matrix[:, 0] = np.ones((final_matrix.shape[0],))
    for j in range(1, degree + 1):
        for k in range(nr_feats):
            final_matrix[:, 1 + k + ((j - 1) * x.shape[1])] = np.power(x[:, k], j)

    if multiply_each:
        for i in range(nr_feats):
            for j in range(nr_feats):
                mul = np.multiply(x[:, i], x[:, j]).reshape((-1, 1))
                final_matrix = np.hstack([final_matrix, mul])
    return final_matrix


def replace_values(config, feats):
    """
    Replace values of -999 with zeros or feature-wise mean.
    :param config:
    :param feats:
    :return:
    """
    if config["replace_with"] == 'mean':
        # Replace -999 values with nan
        feats = np.where(feats == float(-999), np.nan, feats)
        # Column-wise mean on the masked array
        feats = np.where(np.isnan(feats), ma.array(feats, mask=np.isnan(feats)).mean(axis=0), feats)
    elif config["replace_with"] == 'zero':
        feats = np.where(feats == float(-999), float(0), feats)
    return feats


def prepare_train_data(config, args):
    labels, feats, _, feats_name = load_csv_data(config['train_data'])

    # feats = feats[:, [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29]]
    # feats_name = feats_name[[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29]]

    cat_feat_index = np.where(feats_name == 'PRI_jet_num')[0][0]
    # correlation = compute_correlation(feats.T)

    if config["replace_with"] is not None:
        feats = replace_values(config, feats)

    cont_feats = np.delete(feats, cat_feat_index, axis=1)
    cat_feat = feats[:, cat_feat_index]

    if args.see_hist:
        name = 'hist_panel'
        if config["replace_with"] is not None:
            name += '_replaced_with_' + str(config["replace_with"])
        plot_hist_panel(feats, feats_name, config['viz_path'] + name)
        # These seems like good features to me, but selecting only them does not help apparently

    if config["build_poly"]:
        # Build polynomial features
        feats = build_poly(cont_feats, config["degree"], multiply_each=config["multiply_each"])
        feats = np.insert(feats, cat_feat_index + 1, cat_feat, axis=1)
    else:
        # Create x 'tilda'
        feats = build_model_data(feats)
    labels = labels.reshape((labels.shape[0], 1))

    # Feature standardization: we should not standardize feature 23 because it is categorical
    cont_feats = np.delete(feats, cat_feat_index + 1, axis=1)
    feats, tr_mean, tr_std = standardize(cont_feats)
    feats = np.insert(feats, cat_feat_index + 1, cat_feat, axis=1)
    # plot_hist_panel(feats[:, 1:], feats_name, config['viz_path'] + 'hist_panel_after_standardization')

    return feats, labels, tr_mean, tr_std


def prepare_test_data(config, tr_mean, tr_std):
    # Load test data
    _, test_feats, test_index, feats_name = load_csv_data(config['test_data'])
    # test_feats = test_feats[:,
    #              [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29]]
    # feats_name = feats_name[[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29]]

    cat_feat_index = np.where(feats_name == 'PRI_jet_num')[0][0]
    if config["replace_with"] is not None:
        test_feats = replace_values(config, test_feats)

    cont_test_feats = np.delete(test_feats, cat_feat_index, axis=1)
    cat_test_feat = test_feats[:, cat_feat_index]

    if config["build_poly"]:
        # Build polynomial features
        test_feats = build_poly(cont_test_feats, config["degree"], multiply_each=config["multiply_each"])
        test_feats = np.insert(test_feats, cat_feat_index + 1, cat_test_feat, axis=1)
    else:
        # Create x 'tilda'
        test_feats = build_model_data(test_feats)

    # Normalize features
    cont_test_feats = np.delete(test_feats, cat_feat_index + 1, axis=1)
    test_feats, _, _ = standardize(cont_test_feats, tr_mean, tr_std)
    test_feats = np.insert(test_feats, cat_feat_index + 1, cat_test_feat, axis=1)

    return test_feats, test_index


def compute_correlation(x):
    corr = np.corrcoef(x)
    corr = np.tril(corr)
    high_corr = np.where(corr > 0.99)
    high_corr_idxs = [x for x in list(zip(high_corr[0], high_corr[1])) if x[0] != x[1]]
    all_idxs = [item for t in high_corr_idxs for item in t]
    freq = {}
    for i in all_idxs:
        freq[i] = all_idxs.count(i)

    return freq
