import glob
import os
import sys

# This allows for 100 angles per location should we need it
SEPARATOR_MULTIPLIER = 100

def name_hash(filename):
    filename = os.path.basename(filename)
    split = os.path.splitext(filename)[0].split('_')

    if len(split) != 2:
        sys.exit('unrecognised format: {}'.format(filename))

    return int(split[0]) * SEPARATOR_MULTIPLIER + int(split[1])


def load_coordinates(filename):
    if os.path.isfile(filename):
        with open(filename) as fd:
            return fd.read().splitlines()
    else:
        sys.exit('not a file: {}'.format(filename))


def load_images(directory):
    if os.path.isdir(directory):
        files = glob.glob(os.path.join(directory, '*.jpg'))
        return sorted(files, key=name_hash)
    else:
        sys.exit('not a directory: {}'.format(directory))