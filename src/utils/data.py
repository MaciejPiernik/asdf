import tomllib


def load_config(path):
    return tomllib.load(open(path, "rb"))