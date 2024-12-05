import yaml


def get_config(config: str) -> dict:
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def print_debug(*values: object, debug: bool = False) -> None:
    if debug:
        print(*values)