#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" create a list of features """
import sys
import os
import argparse
from collections import OrderedDict

import features
import utils

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "features.txt")
SIGNAL_LENGTH_REDUCE_FACTOR = 80


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", help="output file that contains a list of feature ids", default=DEFAULT_PATH)
    return parser.parse_args(argv)


def create_feature_list():
    """ generate list of features
        
        Args:
        output: output file
        
        Returns:
        None
    """
    # generate test data
    eeg, user, classlabel, name = utils.generate_data()
    
    # only use small fraction of data
    len = features.SIGNAL_LENGTH / SIGNAL_LENGTH_REDUCE_FACTOR
    eeg = eeg[:len, :]

    # generate features
    row = OrderedDict()
    features.extract_all_eeg_features(eeg, row, len, "")

    return row.keys()


def main(argv):
    args = parse_args(argv)
    
    # create feature list
    feature_list = create_feature_list()
    
    # save feature names to an output file
    with open(args.output, "w") as f:
        for name in feature_list:
            f.write(name + "\n")


if __name__ == '__main__':
    exit(main(sys.argv[1:]))

