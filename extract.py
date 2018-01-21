#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" extract features from directories that contain .mat files """
import sys
import os
import traceback
import csv
import argparse
import time
import warnings
import random
import multiprocessing
import numpy as np
import pandas as pd

from collections import OrderedDict

import utils
import create_feature_list
import features

feature_list = None
segment_info = None


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nprocs", help="number of processes used", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--nsegments", help="number of segments", type=int, default=1)
    parser.add_argument("dirs", nargs="+", help="list of raw data directories")
    parser.add_argument("outdir", help="output directory")
    return parser.parse_args(argv)


def is_invalid_instance(eeg, name, i):
    """ check validity of an input EEG signal
        
        Args:
        eeg: input signal
        name: file name
        i: index of a segment
        
        Return:
        None
    """
    nz = np.count_nonzero(eeg)
    if nz == 0:
        warnings.warn("Empty signals across all channels. skipping segment {0} of {1}".format(i, name))
        return True
    return False


def get_segment_id(filename):
    """ return segment id of a file
        
        Args:
        filename: path to a mat file
        
        Return:
        None
    """
    global segment_info
    if segment_info is None:
        segment_info = {}
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "segment.csv"))
        for index, row in df.iterrows():
            segment_info[row["id"]] = str(row["segment"])
    if filename not in segment_info:
        return np.nan
    return segment_info[filename]


def init_with_nans(row):
    """ set nans for all feature values
        
        Args:
        row: a dictionary contains all feature values
        
        Return:
        None
    """
    global feature_list
    
    if feature_list is None:
        feature_list = create_feature_list.create_feature_list()
    
    for feature in feature_list:
        row[feature] = np.nan


def worker(args):
    """ a worker function for processing one EEG signal
        
        Args:
        args: dictionary contains path to input and output files
        
        Return:
        None
    """
    matfile = args["matfile"]
    csvfile = args["csvfile"]
    total = args["total"]
    process_index = args["process_index"]
    nsegments = args["nsegments"]
    
    if os.path.isfile(csvfile) and os.stat(csvfile).st_size > 0:
        print "{0} exist. skipping.".format(os.path.basename(csvfile))
        return

    # load data
    #eeg, user, classlabel, name = utils.parse_matfile(matfile)
    eeg, user, classlabel, name = utils.generate_data()
    rows = []
    segment_len = features.SIGNAL_LENGTH / nsegments
    segment_id = get_segment_id(name)

    for i in range(nsegments):
        row = OrderedDict()

        # get a segment
        eeg_seg = eeg[i*segment_len:(i+1)*segment_len, :]

        # add meta data
        if nsegments == 1:
            row["id"] = name
        else:
            row["id"] = name + "_{0}".format(i)
        row["segment"] = segment_id
        row["user"] = user

        # check validity of signals
        if is_invalid_instance(eeg_seg, name, i):
            set_nans(row)
        else:
            try:
                # extract EEG features
                features.extract_all_eeg_features(eeg_seg, row, segment_len, "")
            except Exception as e:
                init_with_nans(row)
                traceback.print_exc()

        row["classlabel"] = classlabel
        rows.append(row)

    with open(csvfile, "wb") as f:
        print >> f, ",".join([str(v) for v in rows[0].keys()])
        for row in rows:
            print >> f, ",".join([str(v) for v in row.values()])

    sys.stdout.write("processed {0}%, {1}\r".format(round(float(process_index+1)/float(total)*100.0), utils.toc()))
    sys.stdout.flush()


def extract(dirs, outdir, nprocs, nsegments):
    """ go through each data directory and extract features
        
        Args:
        dirs: list of directories to process
        outdir: output directory
        nprocs: number of processes used
        nsegments: number of segments each EEG signal is separated into.
        
        Return:
        None
    """
    for dir in dirs:
        utils.mkdir_p(os.path.join(outdir, os.path.basename(dir)))
        matfiles = utils.list_matfiles(dir)
        worker_args = [{
            "matfile": matfile,
            "csvfile": os.path.join(outdir, os.path.basename(matfile).replace(".mat", ".csv")),
            "total": len(matfiles),
            "process_index": i,
            "nsegments": nsegments,
        } for i, matfile in enumerate(matfiles)]
        utils.tic()
        pool = multiprocessing.Pool(processes=nprocs)
        pool.map(worker, worker_args)


def main(argv):
    args = parse_args(argv)
    extract(args.dirs, args.outdir, args.nprocs, args.nsegments)


if __name__ == '__main__':
    exit(main(sys.argv[1:]))

