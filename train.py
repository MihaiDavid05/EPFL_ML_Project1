import argparse
from utils.config import read_config
from utils.data import load_data, build_model_data

CONFIGS_PATH = 'configs/'
TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    args = parse_arguments()
    config_path = CONFIGS_PATH + args.config_filename + '.yaml'
    c = read_config(config_path)

    # Load and build model data
    feats_name, feats, index, labels = load_data(TRAIN_DATA)
    y, tx = build_model_data(feats, labels)