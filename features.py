#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" functions for calculating EEG features
    
    Given an original EEG signal, this script applies bandpass filters to highlight frequencies that contain descriminative information. Parameters of Morlet (Gabor) filters are selected to roughly emulate the way bandpass filters are applied in [1] by PW Mirowski et al as follows:

    high gamma 65-100Hz
    low gamma 30-55Hz
    high beta 14-30Hz
    low beta 13-15Hz
    alpha 7Hz-13Hz
    theta 4Hz-7Hz
    delta < 4Hz

    Please note according to the reference, alpha to low beta as well as high gamma are the most important frequencies for seizure detection.
    Signals provided in the seizure prediction challenge is about 10 minutes and recorded with 400 Hz sampling rate.
    
    References
    [1] PW Mirowski, Y LeCun, D Madhavan, R Kuzniecky, "Comparing SVM and convolutional networks for epileptic seizure prediction from intracranial EEG", Machine Learning for Signal Processing, 2008
    http://yann.lecun.com/exdb/publis/pdf/mirowski-mlsp-08.pdf
    
    [2] F Lotte, E Miranda, J Castet, "A Tutorial on EEG Signal Processing Techniques for Mental State Recognition in Brain-Computer Interfaces", Guide to Brain-Computer Music Interfacing, Springer, 2015
    https://hal.inria.fr/hal-01055103/file/lotte_EEGSignalProcessing.pdf
    
    [3] F Mormann, T Kreuz, C Rieke, RG Andrzejak, A Kraskov, P David, "On the predictability of epileptic seizures", Clin Neurophysiol , 2005
    http://www.dtic.upf.edu/~ralph/ClinNeurophysiol116569.pdf
    
    [4] R Saab, M.J. McKeown, L.J. Myers, R. Abu-Gharbieh, "A Wavelet Based Approach for the Detection of Coupling in EEG Signals", 2nd International IEEE EMBS Conference on Nueral Engineering, 2005
    http://www.math.ucsd.edu/~rsaab/publications/NER05_1.pdf
"""
import sys, os
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import signal_utils


SAMPLING_RATE = 400
SIGNAL_LENGTH = SAMPLING_RATE * 10 * 60
MORLET_BAND = ["delta", "theta", "alpha", "lowbeta", "highbeta", "lowgamma", "highgamma"]
MORLET_RANGE_MIN = np.array([0, 4,  7, 13, 14, 30,  65])
MORLET_RANGE_MAX = np.array([4, 7, 13, 15, 30, 55, 100])
MORLET_FREQUENCY = (MORLET_RANGE_MIN + MORLET_RANGE_MAX) / 2.0
# Note
# define points of Gaussian envelopes (n * stdev) where min and max of frequency ranges meet.
# TODO this isn't validated in terms of how it's influence on the accuracy
MORLET_NSTDEV = 2.0
MORLET_SIGMA = (np.pi * (MORLET_RANGE_MAX - MORLET_RANGE_MIN) / MORLET_NSTDEV) ** -1.0
MORLET_FILTER_LENGTH = (MORLET_SIGMA * 8 * SAMPLING_RATE).astype(int)
NUM_PAD = np.max(MORLET_FILTER_LENGTH) / 2
morlet_filter_bank = None

GAUSSIAN_SIGMA = MORLET_SIGMA / 8.0 # TODO find more reasonable way to set a standard deviation
gaussian_filter_bank = None

CHANNEL_ID = "0123456789abcdef"


def morlet(f, sigma, fs, t=None):
    """ return a morlet wavelet defined in the time domain (+- 4.0 sigma)
        
        Args:
        f: frequency
        sigma: standard deviation of a Gaussian envelope
        fs: sampling frequency of input signals
        t: time scale
        
        Return:
        a Morlet filter and corresponding time scale
    """
    if t is None:
        t = np.linspace(-sigma*4.0, sigma*4.0, int(sigma*8.0*fs))
    g = np.sqrt(f) * np.exp(2.0j * np.pi * f * t) * np.exp(-0.5 * (t ** 2.0) / sigma ** 2.0)
    return g, t


def gaussian(sigma, fs, t=None):
    """ return a gaussian smoothing filter
        
        Args:
        sigma: standard deviation of a Gaussian envelope
        fs: sampling frequency of input signals
        t: time scale
        
        Return:
        a Gaussian filter and corresponding time scale
    """
    if t is None:
        t = np.linspace(-sigma*4.0, sigma*4.0, int(sigma*8.0*fs))
    gss = np.exp(-0.5 * (t ** 2.0) / sigma ** 2.0)
    gss /= np.sum(gss)
    return gss, t


def init_gaussian_filter_bank():
    """ initialize a Gaussian filter bank
        
        Args:
        None
        
        Return:
        a Gaussian filter bank
    """
    # calculate gaussian filters
    filters = []
    for i, sigma in enumerate(GAUSSIAN_SIGMA):
        g, t = gaussian(sigma, SAMPLING_RATE)
        filters.append(g)
    return filters


def get_gaussian_filter_bank():
    """ return a Gaussian filter bank
        
        Args:
        None
        
        Returns:
        a Gaussian filter bank
    """
    global gaussian_filter_bank
    if gaussian_filter_bank is None:
        gaussian_filter_bank = init_gaussian_filter_bank()
    return gaussian_filter_bank


def init_morlet_filter_bank(segment_len, spectrum=True, num_pad=NUM_PAD):
    """ initialize a Morlet wavelet filter bank
        
        Args:
        segment_len: length of an EEG signal
        spectrum: if True, return filters represented in frequency domain
        num_pad: number of signal padding used to avoid signal overlaps around bondaries
        
        Returns:
        a Morlet wavelet filter bank
    """
    # calculate time domain morlet wavelets
    wavelets = []
    for i, (freq, sigma) in enumerate(zip(MORLET_FREQUENCY, MORLET_SIGMA)):
        g, t = morlet(freq, sigma, SAMPLING_RATE)
        wavelets.append(g)
    if not spectrum:
        return wavelets
    # calculate frequency domain morlet wavelets
    morlet_fb = np.zeros((segment_len + num_pad * 2, len(MORLET_FREQUENCY)), dtype=np.complex128)
    for i, wt in enumerate(wavelets):
        morlet_fb[:, i] = np.fft.fft(wt, segment_len + num_pad * 2)
    return morlet_fb


def fft(eeg, num_pad=NUM_PAD):
    """ return real FFT of an EEG signal. Pad a signal if num_pad is non zero
            
        Args:
        eeg: input EEG signal
        num_pad: number of signal padding used to avoid signal overlaps around bondaries
            
        Returns:
        EEG signal represented in frequency domain
    """
    X = np.zeros((eeg.shape[0] + num_pad * 2, eeg.shape[1]), dtype=np.complex128)
    for i in range(eeg.shape[1]):
        if num_pad > 0:
            X[:, i] = np.fft.fft(np.pad(eeg[:, i], [num_pad, num_pad], mode="constant"))
        else:
            X[:, i] = np.fft.fft(eeg[:, i])
    return X


def wavelet_coef(X, filter_fft, filter_length, num_pad, segment_len):
    """ calculate wavelet coefficients given a filter bank represented in
        frequency domain
        
        Args:
        X: EEG signal represented in frequency domain
        filter_fft: a filter bank represented in frequency domain
        filter_length: length of filters
        num_pad: number of signal padding used to avoid signal overlaps around bondaries
        segment_len: length of an EEG signal
        
        Returns:
        wavelet coefficients, and its autospectrum
    """
    W = np.zeros((segment_len, X.shape[1]), dtype=np.complex128)
    W_autocorr = np.zeros((segment_len, X.shape[1]), dtype=float)
    offset = num_pad-filter_length/2
    for i in range(X.shape[1]):
        wt = X[:, i] * np.conj(filter_fft)
        # raw filter response
        W[:, i] = np.fft.ifft(wt)[offset:offset + segment_len]
        # auto correlation of filter response
        W_autocorr[:, i] = np.real(np.fft.ifft(wt * np.conj(wt))[offset:offset + segment_len])
    return W, W_autocorr


def get_morlet_filter_bank(segment_len):
    """ return a Morlet wavelet filter bank
        
        Args:
        segment_len: length of an EEG signal
        
        Returns:
        2d numpy array
    """
    global morlet_filter_bank
    if morlet_filter_bank is None:
        morlet_filter_bank = init_morlet_filter_bank(segment_len, spectrum=True, num_pad=NUM_PAD)
    return morlet_filter_bank


def maximum_linear_cross_correlation(eeg, mask, row, prefix, fs=400.0, sec=1.0):
    """ extract maximal cross correlation features described in [1]
        
        Args:
        
        eeg: input EEG signal
        mask: mask array that indicates where EEG signal recording had dropped
        row: OrderedDict for storing all features
        prefix: prefix for feature names
        fs: sampling frequency
        sec: duration of a time window used to limit the search

        Returns:
        None
    """
    N, M = eeg.shape
    L = int(fs * sec * 0.5)
    auto_corr = np.array([np.sum(eeg[:, i] ** 2.0) for i in range(M)]) / np.sum(mask, axis=0).astype(float)
    for i in range(M-1):
        xi = eeg[:, i]
        mi = mask[:, i]
        for j in range(i+1, M):
            xj = eeg[:, j]
            mj = mask[:, j]
            c = scipy.signal.fftconvolve(xi, xj[::-1], mode='same')[N/2-L:N/2+L]
            m = scipy.signal.fftconvolve(mi, mj[::-1], mode='same').astype(float)[N/2-L:N/2+L]
            valid = m > (N * 1e-5)
            max_c = np.max(c[valid] / m[valid])
            row[prefix + "max_linear_xcorr_" + CHANNEL_ID[i] + CHANNEL_ID[j]] = max_c / np.sqrt(auto_corr[i] * auto_corr[j])


def spectral(X, row, prefix):
    """ extract band power features [2], i.e., the power of the EEG signal in a specific frequency band
        
        Args:
        X: EEG signals represented in frequency domain
        row: OrderedDict for storing all features
        prefix: prefix for feature names
        
        Returns:
        None
    """
    nchannel = X.shape[1]
    
    # spectrum band power, spectral edge frequency
    power_spectrum = np.square(np.abs(X))
    f = np.fft.fftfreq(X.shape[0], 1/float(SAMPLING_RATE))
    i_f40 = np.argmin(np.abs(f-40.0))
    for i in range(nchannel):
        p = np.sum(power_spectrum[f < MORLET_RANGE_MAX[-1]])
        for k, (r_min, r_max) in enumerate(zip(MORLET_RANGE_MIN, MORLET_RANGE_MAX)):
            sp_bpw = np.nan
            if p > 0.0:
                bpw = np.sum(power_spectrum[(f >= r_min) & (f < r_max)]) / p
            row[prefix + "spectral_bandpower_" + MORLET_BAND[k] + "_" + CHANNEL_ID[i]] = sp_bpw
        p_cumsum = np.cumsum(p)
        sp_edge = np.nan
        if p > 0.0:
            sp_edge = f[np.argmin(np.abs(p_cumsum - power_spectrum[i_f40] * 0.5))]
        row[prefix + "spectral_edge_" + CHANNEL_ID[i]] = sp_edge
        auto_corr = np.real(np.fft.ifft(X[:, i] * np.conj(X[:, i])))
        indices = np.where(np.diff(np.sign(auto_corr)))[0]
        index = len(auto_corr) if len(indices) == 0 else indices[0]

        # auto correlation features calculated over EEG signals represented in frequency domain
        row[prefix + "spectral_autocorr_decay_" + CHANNEL_ID[i]] =  float(index) / float(SAMPLING_RATE) * 1000.0


def wavelet(X, mask, morlet_fb, gaussian_fb, row, segment_len, prefix):
    """ extract all wavelet domain features
        
        Args:
        X: EEG signals represented in frequency domain
        mask: mask array that indicates where EEG signal recording had dropped
        morlet_fb: Morlet wavelet filter bank
        gaussian_fb: Gaussian filter bank
        row: OrderedDict for storing all features
        segment_len: length of an EEG signal
        prefix: prefix for feature names
        
        Returns:
        None
    """
    nchannel = X.shape[1]
    
    for k in range(morlet_fb.shape[1]):
        # calculate wavelet coefficients
        W, W_autocorr = wavelet_coef(X, morlet_fb[:, k], MORLET_FILTER_LENGTH[k], NUM_PAD, segment_len)
        phase = np.angle(W)
        amplitude = np.abs(W)
        auto_spectra = np.square(amplitude)
        
        basename = "wavelet_" + MORLET_BAND[k]
        
        for i in range(nchannel):
            
            # univariate wavelet features
            basic_stats(amplitude[mask[:, i]==1, i], row, prefix + basename, "_" + CHANNEL_ID[i], True, False)
            
            # auto correlation features over wavelet coefficients
            indices = np.where(np.diff(np.sign(W_autocorr[:, i])))[0]
            index = W_autocorr.shape[0] if len(indices) == 0 else indices[0]
            row[prefix + basename + "_autocorr_decay_" + CHANNEL_ID[i]] = float(index) / float(SAMPLING_RATE) * 1000.0

        for i in range(nchannel-1):
            for j in range(i+1, nchannel):
                # bivariate features
                channel_id_pair = "_" + CHANNEL_ID[i] + CHANNEL_ID[j]
                phase_delta = phase[:, i] - phase[:, j]
                mask_union = ((mask[:, i]==1) & (mask[:, j]==1)) == 1
                count = np.sum(mask_union)
                
                # calculate SPLV and correlation coefficients of wavelet coefficients
                splv = np.nan
                if count > 0:
                    splv = np.abs(np.sum(np.exp(1.0j * phase_delta)[mask_union]) / float(count))
                row[prefix + basename + "_splv" + channel_id_pair] = splv
                row[prefix + basename + "_corrcoef" + channel_id_pair] = scipy.stats.pearsonr(amplitude[mask_union, i], amplitude[mask_union, j])[0]

                # extract statistics of phase
                basic_stats(phase_delta[mask_union], row, prefix + basename + "_phase_original", channel_id_pair, False, True)
                basic_stats(np.unwrap(phase_delta)[mask_union], row, prefix + basename + "_phase_unwrapped", channel_id_pair, True, True)
            
                # extract statistics of wavelet coherence
                coherence = wavelet_coherence(W[:, i], W[:, j], auto_spectra[:, i], auto_spectra[:, j], mask_union, gaussian_fb[k])
                basic_stats(coherence, row, prefix + basename + "_coherence", channel_id_pair, False, False)


def wavelet_coherence(wi, wj, wii, wjj, m, gaussian_filter):
    """ extract wavelet coherence features
        
        Args:
        wi: wavelet coefficients of ith channel
        wj: wavelet coefficients of jth channel
        wii: autospectrum of ith channel
        wjj: autospectrum of jth channel
        m: mask array that indicates where EEG signal recording had dropped
        gaussian_filter: a lowpass filter
        
        Returns:
        None
    """
    wij = scipy.signal.fftconvolve(wi * np.conj(wj), gaussian_filter, mode='same')
    return np.abs(wij[m]) / np.sqrt(wii[m] * wjj[m])


def basic_stats(x, row, prefix, suffix, ranges=False, entropy=False):
    """ extract basic statistics
        
        Args:
        x: a list of values
        row: OrderedDict for storing all features
        prefix: prefix for feature names
        range: include min, max, range as features
        entropy: include entropy
        
        Returns:
        None
    """
    x = x[~np.isnan(x)]
    if len(x) == 0:
        row[prefix + "_mean" + suffix] = row[prefix + "_std" + suffix] = \
        row[prefix + "_skew" + suffix] = row[prefix + "_kurtosis" + suffix] = \
        row[prefix + "_entropy" + suffix] = np.nan
        return
    row[prefix + "_mean" + suffix] = np.mean(x)
    row[prefix + "_std" + suffix] = np.std(x, ddof=1)
    row[prefix + "_skew" + suffix] = scipy.stats.skew(x)
    row[prefix + "_kurtosis" + suffix] = scipy.stats.kurtosis(x)
    v_min = np.min(x)
    v_max = np.max(x)
    if ranges:
        row[prefix + "_min" + suffix] = v_min
        row[prefix + "_max" + suffix] = v_max
        row[prefix + "_range" + suffix] = v_max - v_min
    if entropy:
        nbins = int(np.exp(0.626+0.4*np.log(len(x)-1)))
        count, edges = np.histogram(x, bins=nbins, range=(v_min, v_max))
        prob = count / float(np.sum(count))
        row[prefix + "_entropy" + suffix] = (np.log(nbins) - scipy.stats.entropy(prob)) / np.log(nbins)


def get_eeg_mask(eeg):
    """ init an integer array that indicate missing sensor values
        
        Args:
        eeg: input EEG signal
        
        Returns:
        mask array that indicates where EEG signal recording had dropped
    """
    mask = np.zeros(eeg.shape, dtype=int)
    mask[eeg != 0] = 1
    return mask


def preprocess(eeg, mask):
    """ apply lowpass and highpass filters and only
        preserve frequency contents between 0.5 to 120 Hz
        
        Args:
        eeg: input EEG signal with its shape = (L, n_channel). L varies depending
             on nsegments parameter and n_channel is always 16.
        mask: mask array that indicates where EEG signal recording had dropped
        
        Returns:
        None
    """
    for i in range(eeg.shape[1]):
        signal_utils.butter_lowpass_filter(eeg[:, i], 120, SAMPLING_RATE, order=7)
        signal_utils.butter_highpass_filter(eeg[:, i], 0.5, SAMPLING_RATE, order=7)
    eeg[mask == 0] = 0


def extract_all_eeg_features(eeg, row, segment_len, prefix):
    """ extract all EEG features
        
        Args:
        eeg: input EEG signal
        row: OrderedDict for storing all features
        segment_len: length of an EEG signal
        prefix: prefix for feature names
        
        Returns:
        None
    """
    
    # obtain a mask for dropped signals
    mask = get_eeg_mask(eeg)
    
    # preprocess
    preprocess(eeg, mask)

    # extract maximum linear cross correlation
    maximum_linear_cross_correlation(eeg, mask, row, prefix, fs=SAMPLING_RATE, sec=1.0)

    # calculate fft of eeg signals for each channel
    X = fft(eeg, NUM_PAD)
    
    # extract spectral features
    spectral(X, row, prefix)
    
    # extract wavelet related features
    morlet_fb = get_morlet_filter_bank(segment_len)
    gaussian_fb = get_gaussian_filter_bank()
    wavelet(X, mask, morlet_fb, gaussian_fb, row, segment_len, prefix)

