import argparse
from utils.config import read_config
from utils.pipelines import model_all_data, model_by_jet, model_by_jet_and_mmcder

CONFIGS_PATH = '../configs/'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Config name that you want to use during the run.')
    parser.add_argument('--test', action='store_true', help='Also test and create submission file')
    parser.add_argument('--see_hist', action='store_true', help='See features histogram panel')
    parser.add_argument('--see_loss', action='store_true', help='See training loss plot')
    parser.add_argument('--see_pca', action='store_true', help='See PCA with 2 components')
    parser.add_argument('--sub_models_by_jet', type=str, default='0,1,2', help='Choose which sub-model to run. 0-jet'
                                                                               '_zero, 1-jet_one, 2-more than 1 jet')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments and get configurable parameters
    cli_args = parse_arguments()
    config_path = CONFIGS_PATH + cli_args.config_filename + '.yaml'
    config = read_config(config_path)
    output_filename = config['output_path'] + cli_args.config_filename + '_submission'
    by_jet = cli_args.config_filename.split('_')[-1] == '3models'
    by_jet_and_feat = cli_args.config_filename.split('_')[-1] == '6models'
    cli_args.sub_models_by_jet = [int(i) for i in cli_args.sub_models_by_jet.split(',')]

    # If there are 3 subsets, split by jet number, predict on each of them
    if by_jet:
        model_by_jet(cli_args, config, output_filename)
    elif by_jet_and_feat:
        model_by_jet_and_mmcder(cli_args, config, output_filename)
    else:
        model_all_data(cli_args, config, output_filename)

    # We have unbalanced data: 85667 signals, 164333 backgrounds, maybe try class weighted reg
