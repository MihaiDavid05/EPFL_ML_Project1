import csv
import numpy as np
import numpy.ma as ma
from random import randrange
from utils.vizualization import plot_hist_panel, plot_pca, plot_correlation
from utils.algo import predict_labels, get_f1, get_precision_recall_accuracy
from utils.implementations import model


def load_csv_data(data_path, sub_sample=False):
    """
    Loads data.
    :param data_path: Path to data.
    :param sub_sample: Whether to consider only a subsample of the dataset.
    :return: Labels, features, test sample ids and features names
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    feats_name = np.genfromtxt(data_path, delimiter=",", max_rows=1, autostrip=True, dtype='str')[2:]
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (0, 1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0

    # Sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]
    return yb, input_data, ids, feats_name


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd.
    :param ids: Event ids associated with each prediction.
    :param y_pred: Predicted class labels.
    :param name: String name of .csv output file to be created.
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2) if int(r2) == 1 else -1})


def standardize(x, tr_mean=None, tr_std=None):
    """
    Standardize the original data set by substracting mean and dividing by standard deviation.
    :param x: Input data.
    :param tr_mean: Training feature-wise mean (for test data).
    :param tr_std: Training feature-wise standard deviation (for test data).
    :return: Standardized data, training mean and standard deviation
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
    x = append_constant_column(x)

    return x, (mean_x, std_x)


def normalize(x, diff=None, minim=None):
    """
    Apply normalization to data
    :param x: Input data
    :param diff: Training feature-wise max - min (for test data).
    :param minim: Training feature-wise min (for test data).
    :return: Normalized data, training difference and minimum.
    """
    # Get min, max and difference
    if minim is None and diff is None:
        xmin = np.min(x, axis=0)
        xmax = np.max(x, axis=0)
        xdiff = xmax - xmin
    else:
        xmin = minim
        xdiff = diff

    # Apply normalization
    x = (x - xmin) / xdiff

    x = append_constant_column(x)
    return x, (xdiff, xmin)


def log_transform(x, model_key=''):
    """
    Log transformation for positive features.
    :param x: Input features.
    :param model_key: String name if sub_model is used.
    :return: Log-transformed features.
    """
    # Get features indexes that need log transform.

    # if model_key == 'zero_jet':
    #     feats_to_log = [4, 7, 10]
    # elif model_key == 'one_jet':
    #     feats_to_log = [0, 1, 2, 3, 5, 6, 7, 9, 12, 15, 17, 18, 21]
    # elif model_key == 'more_than_one_jet':
    #     feats_to_log = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 22, 25, 28]
    # else:
    #     feats_to_log = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]

    feats_to_log = np.where(np.all(x >= 0, axis=0))[0]
    for j in feats_to_log:
        x[:, j] = np.log(x[:, j] + 1)
    return x


def append_constant_column(x):
    """
    Append a column full of one.
    :param x: Input features.
    :return: Features with first column full of ones.
    """
    tx = np.c_[np.ones(x.shape[0]), x]
    return tx


def split_data(x, y, ratio=0.8, seed=1):
    """
    Split the dataset into training and validation sets.
    :param x: Input data.
    :param y: Labels.
    :param ratio: Ratio of training data used from the entire data set. 1-ratio is used as validation set.
    :param seed: Used for random functions.
    :return: Training data and labels, validation data and labels.
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


def cross_validation_split(data, folds=5):
    """
    Create folds for cross validation.
    :param data: Input data.
    :param folds: Number of folds for cross validation.
    :return:
    """
    data_split = list()
    data_copy = list(data)
    fold_size = int(len(data) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy.pop(index))
        data_split.append(fold)
    return data_split


def build_poly(x, degree, multiply_each=False, square_root=False, trig=None):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: Input data.
    :param degree: Polynomial degree.
    :param multiply_each: Form new columns by multiplying xi * xj for i,j=0,...,nr_feats and i != j
    :param square_root: Take square root of each feature.
    :param trig: Column indices on which to apply cos and sin.
    :return: Expanded data with methods above.
    """
    # Build polynomial features
    nr_feats = x.shape[1]
    final_matrix = np.zeros((x.shape[0], nr_feats + (degree - 1) * nr_feats + 1))
    final_matrix[:, 0] = np.ones((final_matrix.shape[0],))
    for j in range(1, degree + 1):
        for k in range(nr_feats):
            final_matrix[:, 1 + k + ((j - 1) * x.shape[1])] = np.power(x[:, k], j)
    # Build multiplied features
    if multiply_each:
        for i in range(nr_feats):
            for j in range(nr_feats):
                if i != j:
                    mul = np.multiply(x[:, i], x[:, j]).reshape((-1, 1))
                    final_matrix = np.hstack([final_matrix, mul])
    # Build squared root features for positive feature columns
    if square_root:
        cont_poz_feats_idx = np.where(np.all(x >= 0, axis=0))[0]
        for i in cont_poz_feats_idx:
            root = np.sqrt(x[:, i]).reshape((-1, 1))
            final_matrix = np.hstack([final_matrix, root])

    if trig is not None:
        for i in trig:
            final_matrix = np.hstack([final_matrix, np.cos(x[:, i]).reshape((-1, 1)), np.sin(x[:, i]).reshape((-1, 1))])

    return final_matrix


def replace_values(config, x):
    """
    Replace values of -999 with zeros or feature-wise mean/mode/median.
    :param config: Configuration parameters.
    :param x: Input data.
    :return: Data with replaced values by one of the methods above.
    """
    # Replace -999 values with nan
    x = np.where(x == float(-999), np.nan, x)
    if config["replace_with"] == 'mean':
        # Column-wise mean on the masked array
        x = np.where(np.isnan(x), ma.array(x, mask=np.isnan(x)).mean(axis=0), x)
    elif config["replace_with"] == 'zero':
        # Replace nan values with 0
        x = np.where(np.isnan(x), float(0), x)
    elif config["replace_with"] == 'median':
        # Column-wise median on the masked array
        x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    elif config["replace_with"] == 'mode':
        # Column-wise modes
        modes = []
        for j in range(x.shape[1]):
            not_nan = ~np.isnan(x[:, j])
            vals, counts = np.unique(x[:, j][not_nan], return_counts=True)
            modes.append(vals[np.argmax(counts)])
        x = np.where(np.isnan(x), modes, x)
    return x


def prepare_train_data(config, args, y, x, x_name=None, model_key=''):
    """
    Training data pre-processing pipeline.
    :param config: Configuration parameters.
    :param args: Command line arguments provided when run.
    :param y: Labels.
    :param x: Input features.
    :param x_name: Feature names.
    :param model_key: String name if sub_model is used.
    :return: Training data, labels and feature statistics.
    """
    log_y = True
    angle_features = None

    # Gather or columns related to angles
    if config["trig"]:
        angle_features = [i for i in range(len(x_name)) if 'phi' in x_name[i]]

    # Remove samples with outlier features
    if config["remove_outliers"]:
        x, y = remove_outliers(x, y, config["remove_outliers"])

    # Replace -999 values
    if config["replace_with"] is not None:
        x = replace_values(config, x)

    # Visualize features in 2D.
    if args.see_pca:
        compute_pca(x, y, config['viz_path'] + '2_comp_pca_' + model_key)

    # See features histograms panel
    if args.see_hist:
        name = 'train_hist_panel_replaced_with_' + str(config["replace_with"]) + '_' + model_key
        plot_hist_panel(x, x_name, config['viz_path'] + name, log_scale_y=log_y)

    # Apply log transformation to positive features
    if config["log_transform"]:
        x = log_transform(x, model_key)

    if config["build_poly"]:
        # Expand features by polynomial degrees or other methods
        x = build_poly(x, config["degree"], multiply_each=config["multiply_each"],
                       square_root=config["square_root"], trig=angle_features)
    else:
        # If not expanding features, just append column of ones
        x = append_constant_column(x)
    # Reshape labels to match size for training
    y = y.reshape((y.shape[0], 1))

    # Apply normalization or standardization
    if config["only_normalize"]:
        x, stats = normalize(x[:, 1:])
    else:
        x, stats = standardize(x[:, 1:])

    # See features histograms panel after scaling
    if args.see_hist:
        name = 'train_hist_panel_replaced_with_' + str(config["replace_with"]) + '_end_preprocessing_' + model_key
        plot_hist_panel(x[:, 1:], x_name, config['viz_path'] + name, log_scale_y=log_y)

    return x, y, stats


def prepare_test_data(config, s1, s2, x, x_name, model_key='', x_index=None, y=None):
    """
    Test data pre-processing pipeline.
    :param config: Configuration parameters.
    :param s1: Mean or min of training features.
    :param s2: Standard deviation or (max - min) of training features.
    :param x: Input data.
    :param x_name: Features names.
    :param model_key: String name if sub_model is used.
    :param x_index: Test samples indexes.
    :param y: Labels, if set is used for validation instead of testing.
    :return:
    """
    angle_features = None

    # Gather or columns related to angles
    if config["trig"]:
        angle_features = [i for i in range(len(x_name)) if 'phi' in x_name[i]]

    # Replace -999 values with zeros/mean
    if config["replace_with"] is not None:
        x = replace_values(config, x)

    # Apply log transformation to positive features
    if config["log_transform"]:
        x = log_transform(x, model_key)

    if config["build_poly"]:
        # Expand features by polynomial degrees or other methods
        x = build_poly(x, config["degree"], multiply_each=config["multiply_each"],
                       square_root=config["square_root"], trig=angle_features)
    else:
        # If not expanding features, just append column of ones
        x = append_constant_column(x)

    # Feature standardization or normalization
    if config["only_normalize"]:
        x, _ = normalize(x[:, 1:], s1, s2)
    else:
        x, _ = standardize(x[:, 1:], s1, s2)
    # Reshape labels to match size for validation
    if y:
        y = y.reshape((y.shape[0], 1))

    return x, x_index, y


def compute_correlation(x, x_name, output_path=None):
    """
    Compute correlation between features.
    :param x: Input data.
    :param x_name: Features names.
    :param output_path: Plot output path.
    :return: Correlated feature indexes
    """
    corrm = np.corrcoef(x.T)
    plot_correlation(corrm, np.ravel(x_name), output_path=output_path)

    corr1 = corrm - np.diagflat(corrm.diagonal())
    corr = np.tril(corr1)
    high_corr = np.where(corr > 0.9)
    high_corr_idxs = [x for x in list(zip(high_corr[0], high_corr[1])) if x[0] != x[1]]
    all_corr_idxs = list(set([int(item) for t in high_corr_idxs for item in t]))

    return all_corr_idxs


def drop_correlated(data, model_key, config, drop_idxs=None):
    """
    Remove highly correlated features.
    :param data: Input data dictionary.
    :param model_key: Key for data corresponding to one of the subsets.
    :param config: Configurable parameters.
    :param drop_idxs: Feature indexes found in training for dropping them at test time.
    :return: New data dictionary and correlated features indexes.
    """
    if drop_idxs is not None:
        corr_idxs = drop_idxs
    else:
        corr_idxs = compute_correlation(data[model_key][1], data[model_key][-1],
                                        output_path=config['viz_path'] + 'correlation_' + model_key)
        print("Dropped {} correlated features for model {}\n".format(len(corr_idxs), model_key))
    data[model_key][1] = np.delete(data[model_key][1], corr_idxs, axis=1)
    data[model_key][-1] = np.delete(data[model_key][-1], corr_idxs)

    return data, corr_idxs


def remove_outliers(x, y, removal_type='std'):
    """
    Remove samples that have outliers features.
    A sample is considered outlier if at least a sample feature is +-3 standard deviations from the mean of the feature.
    :param x: Input data.
    :param y: Labels.
    :param removal_type: IQR or std based removal ('std' or 'iqr')
    :return:
    """
    outliers = []
    if removal_type == 'std':
        # Compute feature-wise mean and standard deviation.
        mean_1 = np.mean(x, axis=0)
        std_1 = np.std(x, axis=0)
        for i in range(x.shape[0]):
            z_scores_per_sample = (x[i] - mean_1) / (std_1 + 0.0000001)
            # If there is a sample feature with absolute z_score above 3 consider the sample as an outlier
            if np.any(np.abs(z_scores_per_sample) > 3):
                outliers.append(i)
    elif removal_type == 'iqr':
        x = np.where(x == float(-999), np.nan, x)
        q1 = np.nanquantile(x, 0.25, axis=0)
        q3 = np.nanquantile(x, 0.75, axis=0)
        median = np.nanmedian(x, axis=0)
        iqr = q3 - q1
        minim = median - 2.22 * iqr
        maxim = median + 2.22 * iqr

        for i in range(x.shape[0]):
            condition = np.logical_and(~np.isnan(x[i]), np.logical_or(x[i] < minim, x[i] > maxim))
            if np.any(condition):
                outliers.append(i)
        x = np.where(np.isnan(x), float(-999), x)

    # Delete samples with outliers
    x = np.delete(x, outliers, axis=0)
    y = np.delete(y, outliers)
    return x, y


def compute_pca(x, y, output_path):
    """
    Perform PCA with 2 principal components.
    :param x: Scaled input data.
    :param y: Labels.
    :param output_path: Path for plot.
    """
    # Compute covariance matrix and eigen values and vectors
    cov_matrix = np.cov(x.T)
    values, vectors = np.linalg.eig(cov_matrix)

    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))

    # Get projected features
    projected_1 = x.dot(vectors.T[0])
    projected_2 = x.dot(vectors.T[1])

    plot_pca(projected_1, projected_2, np.ravel(y), output_path)


def split_data_by_jet(x, y=None, test_index=None):
    """
    Splits data by PRI_jet_num feature values: nr_jet = 0, nr_jet = 1 or nr_jet > 1
    :param x: Input data.
    :param y: Labels.
    :param test_index: Indexes for test samples.
    :return: New data dictionary. At each key you will find training or testing data and labels.
    """
    # Get samples indexes for each condition (jet number)
    cond_zero = x[:, 22] == 0
    cond_one = x[:, 22] == 1
    cond_two_three = np.logical_or(x[:, 22] == 2, x[:, 22] == 3)
    # Build data dictionary
    data_dict = {"zero_jet": [test_index[cond_zero], x[cond_zero], y[cond_zero]],
                 "one_jet": [test_index[cond_one], x[cond_one], y[cond_one]],
                 "more_than_one_jet": [test_index[cond_two_three], x[cond_two_three], y[cond_two_three]]
                 }
    return data_dict


def split_data_by_ffeat(data_dict):
    """
    Splits data by DER_mass_MMCDER feature values.
    :param data_dict: Dictionary with data split by jet.
    :return: New data dictionary. At each key you will find training or testing data and labels.
    """
    new_data_dict = {}
    for k, v in data_dict.items():
        cond_null = np.where(v[1][:, 0] == -999)[0]
        cond_ok = np.where(v[1][:, 0] != -999)[0]
        new_data_dict[k + '_0'] = [data_dict[k][0][cond_null], data_dict[k][1][cond_null], data_dict[k][2][cond_null]]
        new_data_dict[k + '_1'] = [data_dict[k][0][cond_ok], data_dict[k][1][cond_ok], data_dict[k][2][cond_ok]]

    return new_data_dict


def remove_useless_columns(data_by_jet, feats_names):
    """
    Remove columns full of -999, 0 or nan for each subset given by jet number
    :param data_by_jet: Dictionary with data for each subset given by jet number.
    :param feats_names: Feature names.
    :return: New clean data dictionary with feature names.
    """
    for k, v in data_by_jet.items():
        # Get indexes of columns full of useless values
        bad_columns_idx = np.where(np.all(np.isin(v[1], [-999, 0, 1, 2, 3]), axis=0))[0]
        # Delete this columns and store clean data
        new_x = np.delete(v[1], bad_columns_idx, axis=1)
        new_x_names = [feats_names[i] for i in range(len(feats_names)) if i not in bad_columns_idx]
        data_by_jet[k][1] = new_x
        data_by_jet[k].append(new_x_names)

    return data_by_jet


def do_cross_validation(x, y, c):
    """
    Perform cross validation.
    :param x: Input data.
    :param y: Labels.
    :param c: Config parameters.
    :return: Training and validation metrics
    """
    # Concatenate feats and labels
    data = np.hstack((x, y))
    # Split in k-folds for cross_validation
    folds = np.array(cross_validation_split(data))
    # Initialize metric lists
    final_val_f1, final_train_f1, final_train_acc, final_val_acc = [], [], [], []

    for i, fold in enumerate(folds):
        # Get validation data and labels
        val_feats, val_labels = (fold[:, :-1], fold[:, -1].reshape((-1, 1)))
        # Set training data and labels as the rest
        train_split = np.vstack([fold for j, fold in enumerate(folds) if j != i])
        tr_feats, tr_labels = (train_split[:, :-1], train_split[:, -1].reshape((-1, 1)))
        # Find weights with one of the models
        w, tr_loss = model(tr_labels, tr_feats, c)
        # Make predictions for both training and validation
        tr_preds = predict_labels(w, tr_feats, c["reg_threshold"])
        val_preds = predict_labels(w, val_feats, c["reg_threshold"])
        # Get metrics for training and validation
        tr_f1 = get_f1(tr_preds, tr_labels)
        val_f1 = get_f1(val_preds, val_labels)
        _, _, tr_acc = get_precision_recall_accuracy(tr_preds, tr_labels)
        _, _, val_acc = get_precision_recall_accuracy(val_preds, val_labels)
        # Update metrics
        final_val_f1.append(val_f1)
        final_train_f1.append(tr_f1)
        final_train_acc.append(tr_acc)
        final_val_acc.append(val_acc)

    # Compute and return final metrics
    return np.mean(final_val_f1), np.mean(final_train_f1), np.mean(final_val_acc), np.mean(final_train_acc)
