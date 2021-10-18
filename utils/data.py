import csv
import numpy as np
import numpy.ma as ma
from random import randrange
from utils.vizualization import plot_hist_panel, plot_pca
from utils.algo import predict_labels, get_f1
from utils.implementations import logistic_regression, reg_logistic_regression


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

    return x, (mean_x, std_x)


def normalize(x, diff=None, minim=None):
    """
    Apply normalization to data
    :param x: Input data
    :param diff:
    :param minim:
    :return:
    """
    if minim is None and diff is None:
        xmin = np.min(x, axis=0)
        xmax = np.max(x, axis=0)
        xdiff = xmax - xmin
    else:
        xmin = minim
        xdiff = diff

    x = (x - xmin) / xdiff
    x = append_constant_column(x)

    return x, (xdiff, xmin)


def append_constant_column(feats):
    """
    Get feats "tilda".
    :param feats:
    :return:
    """
    tx = np.c_[np.ones(feats.shape[0]), feats]
    return tx


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
    # Set seed
    np.random.seed(seed)
    # Randomly select indexes for training and validation
    num_row = y.shape[0]
    indices = np.random.RandomState(seed=seed).permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[:index_split]
    index_val = indices[index_split:]
    # Create split
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


def build_poly(x, degree, multiply_each=False, square_root=False):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: input data
    :param degree: polynomial degree
    :param multiply_each: Form new columns by muliplying xi * xj for i,j=0,...,nr_feats
    :param square_root: Take square root of each feature.
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
                if i != j:
                    mul = np.multiply(x[:, i], x[:, j]).reshape((-1, 1))
                    final_matrix = np.hstack([final_matrix, mul])

    if square_root:
        for i in range(nr_feats):
            root = np.sqrt(x[:, i]).reshape((-1, 1))
            final_matrix = np.hstack([final_matrix, root])

    return final_matrix


def replace_values(config, feats):
    """
    Replace values of -999 with zeros or feature-wise mean/mode/median.
    :param config:
    :param feats:
    :return:
    """
    # Replace -999 values with nan
    feats = np.where(feats == float(-999), np.nan, feats)
    if config["replace_with"] == 'mean':
        # Column-wise mean on the masked array
        feats = np.where(np.isnan(feats), ma.array(feats, mask=np.isnan(feats)).mean(axis=0), feats)
    elif config["replace_with"] == 'zero':
        # Replace nan values with 0
        feats = np.where(np.isnan(feats), float(0), feats)
    elif config["replace_with"] == 'median':
        # Column-wise median on the masked array
        feats = np.where(np.isnan(feats), np.nanmedian(feats, axis=0), feats)
    elif config["replace_with"] == 'mode':
        # Column-wise modes
        modes = []
        for j in range(feats.shape[1]):
            not_nan = ~np.isnan(feats[:, j])
            vals, counts = np.unique(feats[:, j][not_nan], return_counts=True)
            modes.append(vals[np.argmax(counts)])
        feats = np.where(np.isnan(feats), modes, feats)
    return feats


def prepare_train_data(config, args, labels, feats, feats_name=None):
    """
    Training data preprocessing pipeline.
    :param config:
    :param args:
    :param labels:
    :param feats:
    :param feats_name:
    :return:
    """
    # Remove samples with outlier features
    if config["remove_outliers"]:
        feats, labels = remove_outliers(feats, labels)

    # Set categorical feature index
    cat_feat_index = 22
    # Replace -999 values with zeros/mean
    if config["replace_with"] is not None:
        feats = replace_values(config, feats)

    # Seprate continuous and categorical features
    cont_feats = np.delete(feats, cat_feat_index, axis=1)
    cat_feat = feats[:, cat_feat_index]

    # See features histograms panel
    if args.see_hist:
        name = 'train_hist_panel'
        if config["replace_with"] is not None:
            name += '_replaced_with_' + str(config["replace_with"])
        plot_hist_panel(feats, feats_name, config['viz_path'] + name)

    if config["build_poly"]:
        # Build polynomial features for continuous ones
        feats = build_poly(cont_feats, config["degree"], multiply_each=config["multiply_each"],
                           square_root=config["square_root"])
        feats = np.insert(feats, cat_feat_index + 1, cat_feat, axis=1)
    else:
        # Create x 'tilda' without polynomial features
        feats = append_constant_column(feats)

    labels = labels.reshape((labels.shape[0], 1))

    # Feature standardization or normalization for continuous features
    cont_feats = np.delete(feats, cat_feat_index + 1, axis=1)
    if config["only_normalize"]:
        feats, stats = normalize(cont_feats[:, 1:])
    else:
        feats, stats = standardize(cont_feats)

    # Visualize features in 2D.
    if args.see_pca:
        compute_pca(feats, labels, config['viz_path'] + '2_comp_pca')

    feats = np.insert(feats, cat_feat_index + 1, cat_feat, axis=1)

    # See features histograms panel after scaling
    if args.see_hist:
        plot_hist_panel(feats[:, 1:], feats_name, config['viz_path'] + 'train_hist_panel_after_scale')

    return feats, labels, stats


def prepare_test_data(config, stat1, stat2, test_feats, test_index=None, test_labels=None):
    """
    Test data preprocessing pipeline.
    :param config:
    :param stat1:
    :param stat2:
    :param test_feats:
    :param test_index:
    :param test_labels:
    :return:
    """
    # Set categorical feature index
    cat_feat_index = 22
    # Replace -999 values with zeros/mean
    if config["replace_with"] is not None:
        test_feats = replace_values(config, test_feats)

    # Seprate continuous and categorical features
    cont_test_feats = np.delete(test_feats, cat_feat_index, axis=1)
    cat_test_feat = test_feats[:, cat_feat_index]

    if config["build_poly"]:
        # Build polynomial features for continuous ones
        test_feats = build_poly(cont_test_feats, config["degree"], multiply_each=config["multiply_each"])
        test_feats = np.insert(test_feats, cat_feat_index + 1, cat_test_feat, axis=1)
    else:
        # Create x 'tilda' without polynomial features
        test_feats = append_constant_column(test_feats)

    # Feature standardization or normalization for continuous features
    cont_test_feats = np.delete(test_feats, cat_feat_index + 1, axis=1)
    if config["only_normalize"]:
        test_feats, _ = normalize(cont_test_feats[:, 1:], stat1, stat2)
    else:
        test_feats, _ = standardize(cont_test_feats, stat1, stat2)

    test_feats = np.insert(test_feats, cat_feat_index + 1, cat_test_feat, axis=1)

    if test_labels:
        test_labels = test_labels.reshape((test_labels.shape[0], 1))

    return test_feats, test_index, test_labels


def compute_correlation(x):
    """
    Compute correlation between features.
    :param x: Input data
    :return: A dict with highly correlated feature ids and their number of occurrences.
    """
    corr = np.corrcoef(x)
    corr = np.tril(corr)
    high_corr = np.where(corr > 0.99)
    high_corr_idxs = [x for x in list(zip(high_corr[0], high_corr[1])) if x[0] != x[1]]
    all_idxs = [item for t in high_corr_idxs for item in t]
    freq = {}
    for i in all_idxs:
        freq[i] = all_idxs.count(i)

    return freq


def remove_outliers(feats, labels):
    """
    Remove samples that have outliers features.
    More exactly, remove samples where at least a feature is bigger or lower than 2.22 quantiles from the median of it.
    :param feats:
    :param labels:
    :return:
    """
    outliers = []
    mean_1 = np.mean(feats, axis=0)
    std_1 = np.std(feats, axis=0)
    for i in range(feats.shape[0]):
        z_scores_per_sample = (feats[i] - mean_1) / (std_1 + 0.0000001)
        # If there is a feature with z_score above threshold consider the sample as an outlier ?!?!
        if np.any(np.abs(z_scores_per_sample) > 3):
            outliers.append(i)

    feats = np.delete(feats, outliers, axis=0)
    labels = np.delete(labels, outliers)
    return feats, labels

    # TODO: this didn't work better, why ?!?
    # outliers = []
    # feats = np.where(feats == float(-999), np.nan, feats)
    # q1 = np.nanquantile(feats, 0.25, axis=0)
    # q3 = np.nanquantile(feats, 0.75, axis=0)
    # median = np.nanmedian(feats, axis=0)
    # iqr = q3 - q1
    # minim = median - 2.22 * iqr
    # maxim = median + 2.22 * iqr
    #
    # for i in range(feats.shape[0]):
    #     condition = np.logical_and(~np.isnan(feats[i]), np.logical_or(feats[i] < minim, feats[i] > maxim))
    #     if np.any(condition):
    #         outliers.append(i)
    #
    # feats = np.delete(feats, outliers, axis=0)
    # feats = np.where(np.isnan(feats), float(-999), feats)
    # labels = np.delete(labels, outliers)
    # return feats, labels


def compute_pca(scaled_x, y, output_path):
    """
    Perform PCA with 2 principal components.
    :param scaled_x: Scaled input data
    :param y: Labels
    :param output_path:
    :return:
    """
    cov_matrix = np.cov(scaled_x.T)
    values, vectors = np.linalg.eig(cov_matrix)

    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))

    projected_1 = scaled_x.dot(vectors.T[0])
    projected_2 = scaled_x.dot(vectors.T[1])

    plot_pca(projected_1, projected_2, np.ravel(y), output_path)


def split_data_by_jet(x):
    """
    Splits data by nr_jet=0, nr_jet=1 and nr_jet>1
    :param x:
    :return:
    """
    data_dict = {"zero_jet": x[x[:, 22] == 0],
                 "one_jet": x[x[:, 22] == 1],
                 "more_than_one_jet": x[np.logical_or(x[:, 22] == 2, x[:, 22] == 3)]
                 }
    return data_dict


def do_cross_validation(feats, labels, lambda_, config):
    """
    Perform cross validation.
    :param feats:
    :param labels:
    :param lambda_:
    :param config:
    :return:
    """
    # Concatenate feats and labels
    data = np.hstack((feats, labels))
    # Split in k-folds for cross_validation
    folds = np.array(cross_validation_split(data))

    final_val_f1 = 0
    final_train_f1 = 0
    for i, fold in enumerate(folds):
        # Get validation data and labels for the validation fold
        val_feats, val_labels = (fold[:, :-1], fold[:, -1].reshape((-1, 1)))
        # Set training data and labels as the rest of the folds
        train_split = np.vstack([fold for j, fold in enumerate(folds) if j != i])
        tr_feats, tr_labels = (train_split[:, :-1], train_split[:, -1].reshape((-1, 1)))
        # Find weights
        if lambda_:
            weights, tr_loss = reg_logistic_regression(tr_labels, tr_feats, lambda_, np.zeros((val_feats.shape[1], 1)),
                                                       config['max_iters'], config['gamma'])
        else:
            weights, tr_loss = logistic_regression(tr_labels, tr_feats, np.zeros((val_feats.shape[1], 1)),
                                                   config['max_iters'], config['gamma'])
        # Make predictions for both training and validation
        tr_preds = predict_labels(weights, tr_feats, config["reg_threshold"])
        val_preds = predict_labels(weights, val_feats, config["reg_threshold"])
        # Get f1 score for training and validation
        tr_f1 = get_f1(tr_preds, tr_labels)
        val_f1 = get_f1(val_preds, val_labels)

        final_val_f1 += val_f1
        final_train_f1 += tr_f1
        print("For fold {}, training f1 score is {:.2f} % and validation f1 score is {:.2f} %".format(i + 1,
                                                                                                      tr_f1 * 100,
                                                                                                      val_f1 * 100))
    # Compute final validation accuracy
    return final_val_f1 / len(folds), final_train_f1 / len(folds)
