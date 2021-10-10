from helpers.config import read_config
import argparse

CONFIGS_PATH = 'configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config file that you want to use during the run.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config_path = CONFIGS_PATH + args.config_filename + '.yaml'
    c = read_config(config_path)

    print(c)
