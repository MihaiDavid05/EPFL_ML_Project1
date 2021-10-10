import argparse
import numpy as np
from utils.config import read_config
from utils.data import load_csv_data, create_csv_submission, predict_labels, build_model_data, split_data, standardize
from utils.algo import accuracy
from implementations import logistic_regression

CONFIGS_PATH = 'configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('output_filename', type=str, help='Name of the submission file')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    args = parse_arguments()
    config_path = CONFIGS_PATH + args.config_filename + '.yaml'
    c = read_config(config_path)

    # Load data
    labels, feats, index, feats_name = load_csv_data(c['train_data'])
    # Output data
    output_filename = c['output_path'] + args.output_filename

    # Create x 'tilda' and split data into training and validation
    feats = build_model_data(feats)
    labels = labels.reshape((labels.shape[0], 1))
    # Feature standardization
    feats, mean, std = standardize(feats)
    tr_feats, val_feats, tr_labels, val_labels = split_data(feats, labels, c["ratio"])

    # Train and get weights
    # TODO: maybe change initialization
    weights, tr_loss = logistic_regression(tr_labels, tr_feats, np.zeros((31, 1)), c['max_iters'], c['gamma'])

    tr_preds = predict_labels(weights, tr_feats)
    val_preds = predict_labels(weights, val_feats)

    tr_acc = accuracy(tr_preds, tr_labels)
    val_acc = accuracy(val_preds, val_labels)
    print("Training accuracy is {:.2f} %".format(tr_acc * 100))
    print("Validation accuracy is {:.2f} %".format(val_acc * 100))
    # Dummy submission with the labels

    # TODO: submission on test set
    # Load data
    _, test_feats, test_index, _ = load_csv_data(c['test_data'])
    test_feats = build_model_data(test_feats)
    # Standardize
    test_feats = test_feats - mean
    test_feats = test_feats / (std + 0.0000001)
    # Predict
    test_preds = predict_labels(weights, test_feats)
    # Submit
    create_csv_submission(test_index, test_preds, output_filename)
