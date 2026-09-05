import sys
import yaml
from dacite import from_dict, Config


def load_config_from_file(config_file: str, data_class: type, strict: bool = True):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return from_dict(data_class=data_class, data=config, config=Config(strict=strict))


def preload_config_file_from_args(data_class: type):
    """
    Preload the config file from the command line arguments

    Usage:
    renderformer train rf1 --config_file config.yaml xxx xxx xxx

    Returns:
        dict: the loaded config
    """
    # pop out the first "-u" for scalene
    if sys.argv[0] == '-u':
        sys.argv.pop(0)
    if '--config_file' in sys.argv:
        index = sys.argv.index('--config_file')
        if index + 1 >= len(sys.argv):
            raise ValueError('--config_file requires a path')
        config_file = sys.argv[index + 1]
        config = load_config_from_file(config_file, data_class)
        del sys.argv[index:index + 2]
        return config

    return None
