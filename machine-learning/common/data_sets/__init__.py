from os import path, getcwd

def get_dataset_path(name, extension="csv"):
    return path.join(getcwd(), 'common', 'data_sets', f'{name}.{extension}')