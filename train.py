import argparse
import numpy as np
from utils.config import read_config
from utils.data import load_csv_data, create_csv_submission, predict_labels, build_model_data
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

    # Train and get weights
    feats = build_model_data(feats)
    labels = labels.reshape((labels.shape[0], 1))
    weights, loss = logistic_regression(labels, feats, np.zeros((31, 1)), c['max_iters'], c['gamma'])

    preds = predict_labels(weights, feats)

    acc = accuracy(preds, labels)
    print("Accuracy is {:.2f} %".format(acc * 100))
    # Dummy submission with the labels
    create_csv_submission(index, preds, output_filename)
