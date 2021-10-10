import argparse
from utils.config import read_config
from utils.data import load_csv_data, create_csv_submission

CONFIGS_PATH = 'configs/'
TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'
OUTPUT_PATH = 'submissions/'


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
    labels, feats, index, feats_name = load_csv_data(TRAIN_DATA)
    output_filename = OUTPUT_PATH + args.output_filename

    # Output data
    create_csv_submission(index, labels, output_filename)
