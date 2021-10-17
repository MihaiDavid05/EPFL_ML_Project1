import argparse
from src.run import read_config
from utils.data import load_csv_data
from utils.optimizations import find_best_poly_lambda, find_best_poly_lambda_cross_val, find_best_reg_threshold

CONFIGS_PATH = '../configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    cli_args = parse_arguments()
    config_path = CONFIGS_PATH + cli_args.config_filename + '.yaml'
    c = read_config(config_path)
    labels, feats, _, feats_name = load_csv_data(c['train_data'])

    # Optimizations
    # find_best_poly_lambda(feats, labels, c, cli_args, 'bias_variance')
    # find_best_reg_threshold(feats, labels, c, cli_args)
    find_best_poly_lambda_cross_val(feats, labels, c, cli_args)
