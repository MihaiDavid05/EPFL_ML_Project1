import yaml


def read_config(config_file_path):
    """
    Read config file.
    :param config_file_path: File path to the config.
    :return: Config file as a dict.
    """
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
