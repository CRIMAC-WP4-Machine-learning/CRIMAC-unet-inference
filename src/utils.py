import os
import yaml


def read_config(config_path):
    """
    Read config yaml file
    """
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    # Read the config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

        # Print description
        print(config['description'])

    return config


class CombineFunctions():
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, *args):
        out = args
        for f in self.functions:
            out = f(*out)
        return out