import argparse
import numpy as np
from utils.config import read_config
from utils.data import create_csv_submission, prepare_train_data, prepare_test_data, load_csv_data,\
    do_cross_validation, split_data_by_jet, remove_useless_columns
from utils.algo import predict_labels, get_f1, get_precision_recall_accuracy
from utils.implementations import logistic_regression, reg_logistic_regression, ridge_regression
from utils.vizualization import plot_loss

CONFIGS_PATH = '../configs/'

MODELS = {'reg_log': reg_logistic_regression,
          'log': logistic_regression,
          'ridge': ridge_regression}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('--test', action='store_true', help='Also test and create submission file')
    parser.add_argument('--by_jet', action='store_true', help='Train and test 3 different models on 3 different'
                                                              ' datasets representing the entire dataset split by'
                                                              ' jet number')
    parser.add_argument('--see_hist', action='store_true', help='See features histogram panel')
    parser.add_argument('--see_loss', action='store_true', help='See training loss plot')
    parser.add_argument('--see_pca', action='store_true', help='See PCA with 2 components')

    return parser.parse_args()


def train(config, args, y, x, x_name):
    """
    Pipeline for training.
    :param config: Configuration parameters.
    :param args: Command line arguments
    :param y:
    :param x:
    :param x_name:
    :return:
    """
    # Prepare data for training
    x, y, stat = prepare_train_data(config, args, y, x, x_name)
    # Perform cross validation
    if config['cross_val']:
        final_val_f1, _, final_val_acc, _ = do_cross_validation(x, y, config['lambda'], config)
        print("Validation f1 score is {:.2f} % and accuracy is {:.2f} %".format(final_val_f1 * 100, final_val_acc * 100))
    # Use logisitc regression or regularized logisitc regression and find weights
    if config['lambda'] is not None:
        if config['model'] == 'ridge':
            weights, tr_loss = ridge_regression(y, x, config['lambda'])
        else:
            weights, tr_loss = reg_logistic_regression(y, x, config['lambda'], np.zeros((x.shape[1], 1)),
                                                       config['max_iters'], config['gamma'])
    else:
        weights, tr_loss = logistic_regression(y, x, np.zeros((x.shape[1], 1)), config['max_iters'],
                                               config['gamma'])
    # Plot training loss
    if args.see_loss:
        output_path = config["viz_path"] + 'loss_plot_' + args.config_filename
        plot_loss(range(config['max_iters']), np.ravel(tr_loss), output_path=output_path)

    # Get predictions
    tr_preds = predict_labels(weights, x, config["reg_threshold"])
    # Get F1 score for training
    f1_score = get_f1(tr_preds, y)
    prec, recall, acc = get_precision_recall_accuracy(tr_preds, y)
    print("Training F1 score is {:.2f} % and accuracy is {}".format(f1_score * 100, acc * 100))

    return stat, weights


def test(config, stat1, stat2, tr_weights, output, x, ind):
    """
    Pipeline for testing.
    :param config: Configuration parameters.
    :param stat1: Feature-wise training mean or max - min
    :param stat2: Feature-wise training standard deviation or min
    :param tr_weights: Weights of the trained model.
    :param output: Output filename.
    :param x:
    :param ind:
    :return:
    """
    # Prepare data for testing
    x, ind, _ = prepare_test_data(config, stat1, stat2, x, ind)
    # Get predictions
    y = predict_labels(tr_weights, x, config["reg_threshold"])
    # Create submission file
    create_csv_submission(ind, y, output)


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    cli_args = parse_arguments()
    config_path = CONFIGS_PATH + cli_args.config_filename + '.yaml'
    c = read_config(config_path)
    output_filename = c['output_path'] + cli_args.config_filename + '_submission'

    if cli_args.by_jet:
        y_tr, x_tr, _, x_name_tr = load_csv_data(c['train_data'])
        _, x_te, index_te, _ = load_csv_data(c['test_data'])

        data_dict_tr = split_data_by_jet(x_tr, y_tr)
        data_dict_te = split_data_by_jet(x_te, np.zeros(x_te.shape[0]))

        data_dict_tr = remove_useless_columns(data_dict_tr)
        data_dict_te = remove_useless_columns(data_dict_te)

        for k in data_dict_tr.keys():
            indices_tr, x_tr, y_tr = data_dict_tr[k]
            indices_te, x_te, _ = data_dict_te[k]

            stats_tr, w_tr = train(c, cli_args, y_tr, x_tr, x_name_tr)
            test(c, stats_tr[0], stats_tr[1], w_tr, output_filename, x_te, index_te)
    else:
        # Load data
        labels, feats, _, feats_name = load_csv_data(c['train_data'])
        # Train pipeline
        stats, w = train(c, cli_args, labels, feats, feats_name)
        if cli_args.test:
            # Load data
            _, test_feats, test_index, _ = load_csv_data(c['test_data'])
            # Test pipeline
            test(c, stats[0], stats[1], w, output_filename, test_feats, test_index)

    # TODO: 1. check if multiply_each, in build_poly, helps
    # TODO: 2. visualize val loss and train loss together
    # TODO: 3. make 3 separate models, by jet - check experiment 21

    # TODO maybe: We have an unbalanced dataset: 85667 signals, 164333 backgrounds, try class weighted reg
    # https://machinelearningmastery.com/cost-sensitive-logistic-regression/
