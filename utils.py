#!/usr/bin/python
import sys, os, argparse
import numpy as np
import pandas as pd
import random
import errno
from scipy import sparse
import time
import scipy.io
import features

def mkdir_p(path):
    """
        a function equivalent to "mkdir -p" in bash scripting
        https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
        
        Args:
        path: path to a new directry
        
        Returns:
        None
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


_start_time = time.time()


def tic():
    """ start a timer
        Args:
        None
        
        Return:
        None
    """
    global _start_time
    _start_time = time.time()


def toc():
    """ report an elapsed time since tic() is called
        Args:
        None
        
        Return:
        a string contains a reprot
    """
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    return "time elapsed: {} hr, {} min, {} sec".format(int(t_hour),int(t_min),int(t_sec))


def list_matfiles(path):
    """ list all files ends with .mat
        
        Args:
        path: path to a root directory
        
        Returns:
        list of file names
    """
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(".mat")]


def parse_matfile(path):
    """ read a .mat file and parse its contents
        
        Args:
        path: path to a file
        
        Returns:
        EEG signal, user name, class label, and file name
    """
    data = scipy.io.loadmat(path)["dataStruct"]
    eeg = data[0][0][0]
    name = os.path.basename(path)
    substr = name.replace("new_", "").replace("old_", "").split('.')[0].split('_')
    user = int(substr[0])
    classlabel = np.nan if len(substr) == 2 else int(substr[2])
    return eeg, user, classlabel, name


def generate_data(drop_signal_rate=0.0):
    """ generate an array with random numbers to emulate an EEG signal
        
        Args:
        None
        
        Returns:
        EEG signal, user name, class label, and file name
        """
    eeg = np.random.randint(low=-100, high=100, size=(features.SIGNAL_LENGTH, 16))
    
    if drop_signal_rate > 0.0:
        # intentionally set zeros to a part of EEG signal
        len = int(features.SIGNAL_LENGTH * drop_signal_rate)
        start_index = np.random.randint(0, features.SIGNAL_LENGTH - len)
        eeg[start_index:start_index+len] = 0
    name = "testdata"
    substr = "substr"
    user = "testuser"
    classlabel = 1
    return eeg, user, classlabel, name

