import argparse
import numpy as np
from utils.config import read_config
from utils.data import load_csv_data, create_csv_submission, build_model_data, split_data, normalize,\
    cross_validation_split
from utils.algo import accuracy, do_cross_validation, predict_labels
from implementations import logistic_regression
from utils.vizualization import plot_hist_panel

CONFIGS_PATH = 'configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('--see_hist', action='store_true', help='See features histogram panel')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    args = parse_arguments()
    config_path = CONFIGS_PATH + args.config_filename + '.yaml'
    c = read_config(config_path)
    output_filename = c['output_path'] + args.config_filename + '_submission'

    # Load data
    labels, feats, index, feats_name = load_csv_data(c['train_data'])

    if args.see_hist:
        # TODO: Maybe get rid of useless features by analyzing this plot
        plot_hist_panel(feats, feats_name)
        # These seems like good features to me
        feats = feats[:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29]]

    # Create x 'tilda' and make labels 2D-array
    feats = build_model_data(feats)
    labels = labels.reshape((labels.shape[0], 1))

    # Feature normalization
    feats, mean, std = normalize(feats)

    if c['cross_val']:
        # Concatenate feats and labels
        data = np.hstack((feats, labels))
        # Split in k-folds for cross_validation
        folds = cross_validation_split(data)
        # Get the mean validation accuracy after running cross_validation
        val_acc = do_cross_validation(folds, logistic_regression, c)
        print("Validation accuracy is {:.2f} %".format(val_acc * 100))

    # Train on whole data set and find weights
    weights, tr_loss = logistic_regression(labels, feats, np.zeros((feats.shape[1], 1)), c['max_iters'], c['gamma'])

    # Load test data
    _, test_feats, test_index, _ = load_csv_data(c['test_data'])
    test_feats, _ = build_model_data(test_feats)
    # Normalize features
    test_feats = test_feats - mean
    test_feats = test_feats / (std + 0.0000001)
    # Predictions
    test_preds = predict_labels(weights, test_feats)
    # Create submission file
    create_csv_submission(test_index, test_preds, output_filename)
