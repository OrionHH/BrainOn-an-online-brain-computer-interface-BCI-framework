# -*- coding:utf-8 -*-
'''
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2020.04.23
version 1.0

Application: Base preprocessing framework for building offline model.
'''

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import signal

class BasePreProcessing(metaclass=ABCMeta):
    """

    Parameters
    ----------
    filepath: str, data filepath.
    t_begin: float, unit: second
    t_end: float, unit: second
    fs_down: int
        the down-sampling rate (default: 250Hz).
    chs: list | str, selected channels
    n_classes: int,
        the number of classes.

    """
    CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
        'F1','FZ','F2','F4','F6','F8','FT7','FC5',
        'FC3','FC1','FCZ','FC2','FC4','FC6','FC8','T7',
        'C5','C3','C1','CZ','C2','C4','C6','T8',
        'M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4',
        'CP6','TP8','M2','P7','P5','P3','P1','PZ',
        'P2','P4','P6','P8','PO7','PO5','PO3','POZ',
        'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
    ] # M1: 33. M2: 43.

    def __init__(self, filepath, t_begin, t_end, fs_down=250, chs=None, n_classes=None):

        self.filepath = filepath
        self.fs_down = fs_down
        self.t_begin = t_begin
        self.t_end = t_end
        self.chs = chs
        self.n_classes = n_classes

    @abstractmethod
    def load_data(self):
        """Load data and selected channels by chs.
        """
        pass

    @abstractmethod
    def resample_data(self, raw_data):
        pass

    def _get_iir_sos_band(self, w_pass, w_stop):
        """Get second-order sections (like 'ba') of Chebyshev type I filter for band-pass.

        Parameters
        ----------
        w_pass: list, 2 elements, e.g. [5, 70]
        w_stop: list, 2 elements, e.g. [3, 72]
        args: tuple. Only first element is valid. The other elements can also be developed to extend the functions.
            args[0]: int, down-sampling frequency, unit: Hz.

        Returns
        -------
        sos_system:
            i.e the filter coefficients.
        """
        if len(w_pass) !=2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        wp = [2*w_pass[0]/self.fs_down, 2*w_pass[1]/self.fs_down]
        ws = [2*w_stop[0]/self.fs_down, 2*w_stop[1]/self.fs_down]
        gpass = 4  # it's -3dB when setting as 3.
        gstop = 30  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

        return sos_system

    @abstractmethod
    def filtered_data_iir(self, w_pass_2d, w_stop_2d, data):
        """Filter data using IIR filters.

        Parameters
        ----------
        w_pass_2d: ndarray of shape 2-D,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.

        w_stop_2d: ndarray of shape 2-D,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
            e.g.
            w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
            w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])

        data: ndarray of shape, any shape.
        """
        pass

    def ext_epochs(self, filtered_data, event):
        """Extract epochs according to self.t_begin and self.t_end.

        Parameters
        ----------
        filtered_data: ndarray of shape

        Returns
        -------
        all_data: ndarray of shape
            all data filtered and extracted epochs.
        """
        if self.t_begin is None or self.t_end is None:
            raise ValueError('t_begin or t_end must be set.')

        fs = self.fs_down
        self.all_data = filtered_data[:, int(np.round(fs*self.t_begin)):int(np.round(fs*self.t_end)+1), :]

