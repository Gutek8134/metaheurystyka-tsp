import argparse
from os.path import isfile

def file_path(path: str):
    if isfile(path):
        return True
    raise argparse.ArgumentTypeError("Path is not a file")