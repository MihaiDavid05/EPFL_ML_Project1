import yaml


def read_config(config_file_path='configs/first_experiment.yaml'):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
