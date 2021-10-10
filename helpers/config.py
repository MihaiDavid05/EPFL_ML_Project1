import yaml


def read_config(config_file_path='configs/first_experiment.yaml'):
    """
    Read config file.
    :param config_file_path:
    :return:
    """
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
