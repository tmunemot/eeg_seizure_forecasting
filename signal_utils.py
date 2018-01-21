#!/usr/bin/env python
""" a collection of functions for reading EEG signals and applying basic filtering operations
"""
import os
import scipy.io
import numpy as np
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    """ wrapper for calculating parameters of a lowpass filter
        
        Args:
        cutoff: cutoff frequency
        fs: sampling frequency
        order: order of butterworth filter
        
        Returns:
        parameters used in lowpass filtering
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """ apply a lowpass filter
        
        Args:
        data: input signal
        cutoff: cutoff frequency
        fs: sampling frequency
        order: order of butterworth filter
        
        Returns:
        output signal
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    """ wrapper for calculating parameters of a highpass filter
        
        Args:
        cutoff: cutoff frequency
        fs: sampling frequency
        order: order of butterworth filter
        
        Returns:
        parameters used in highpass filtering
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """ apply a highpass filter
        
        Args:
        data: input signal
        cutoff: cutoff frequency
        fs: sampling frequency
        order: order of butterworth filter
        
        Returns:
        output signal
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    """ wrapper for calculating parameters of a bandpass filter
        
        Args:
        lowcut: cutoff frequency for lowpass
        highcut: cutoff frequency for highpass
        fs: sampling frequency
        order: order of butterworth filter
        
        Returns:
        parameters used in bandpass filtering
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
