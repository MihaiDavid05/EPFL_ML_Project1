import yaml


def read_config(config_file_path):
    """
    Read config file.
    :param string config_file_path: File path to the config.
    :return dict: the config file
    """
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
