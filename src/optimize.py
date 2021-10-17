from src.run import parse_arguments, read_config, CONFIGS_PATH
from utils.data import load_csv_data
from utils.optimizations import find_best_poly


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    cli_args = parse_arguments()
    config_path = CONFIGS_PATH + cli_args.config_filename + '.yaml'
    c = read_config(config_path)
    output_filename = c['output_path'] + cli_args.config_filename + '_submission'
    labels, feats, _, feats_name = load_csv_data(c['train_data'])
    find_best_poly(feats, labels, c, cli_args, 'bias_variance')
