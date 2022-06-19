# -*- coding:utf-8 -*-
"""
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2022-03-14
update: 2022-06
Version 1.1

Application:
    Simulate the online processing flow for program validity that not need connect amplifier or bulid hardware platform.
"""

import os, sys
import warnings
import time

import numpy as np
import joblib
from scipy import signal
from sklearn import metrics
import mne
from mne.io import concatenate_raws
from mne import Epochs

from BaseFramework import BaseProcessingRecog


warnings.filterwarnings('ignore')  # or warnings.filterwarnings('ignore')

symbol_216 = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1', 'Q1', 'R1',
    'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2', 'J2', 'K2', 'L2', 'M2', 'N2', 'O2', 'P2', 'Q2', 'R2',
    'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'I3', 'J3', 'K3', 'L3', 'M3', 'N3', 'O3', 'P3', 'Q3', 'R3',
    'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4', 'I4', 'J4', 'K4', 'L4', 'M4', 'N4', 'O4', 'P4', 'Q4', 'R4',
    'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'I5', 'J5', 'K5', 'L5', 'M5', 'N5', 'O5', 'P5', 'Q5', 'R5',
    'A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6', 'I6', 'J6', 'K6', 'L6', 'M6', 'N6', 'O6', 'P6', 'Q6', 'R6',
    'S1', 'T1', 'U1', 'V1', 'W1', 'X1', 'Y1', 'Z1', '01', '11', '21', '31', '41', '51', '61', '71', '81', '91',
    'S2', 'T2', 'U2', 'V2', 'W2', 'X2', 'Y2', 'Z2', '02', '12', '22', '32', '42', '52', '62', '72', '82', '92',
    'S3', 'T3', 'U3', 'V3', 'W3', 'X3', 'Y3', 'Z3', '03', '13', '23', '33', '43', '53', '63', '73', '83', '93',
    'S4', 'T4', 'U4', 'V4', 'W4', 'X4', 'Y4', 'Z4', '04', '14', '24', '34', '44', '54', '64', '74', '84', '94',
    'S5', 'T5', 'U5', 'V5', 'W5', 'X5', 'Y5', 'Z5', '05', '15', '25', '35', '45', '55', '65', '75', '85', '95',
    'S6', 'T6', 'U6', 'V6', 'W6', 'X6', 'Y6', 'Z6', '06', '16', '26', '36', '46', '56', '66', '76', '86', '96']

CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
    'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
    'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FC8', 'T7',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]  # M1: 33. M2: 43.
# CHANNELS.remove('M1')
# CHANNELS.remove('M2')


class ProcessingRecog():

    num_commands = 216  # 54 commands

    def __init__(self, n_rounds, t_begin_cls, t_end_cls, fs_down, CHANNELS, chs_used, w_pass_2d, w_stop_2d, fs_orig,
                 t_begin_buffer, chs_p3, w_pass_p3=None, w_stop_p3=None, fs_down_p3=None, t_begin_p3=None, t_end_p3=None):

        self.n_rnds = n_rounds
        self.t_begin_cls = t_begin_cls
        self.t_end_cls = t_end_cls
        self.fs_down = fs_down
        self._CHANNELS = CHANNELS
        self.chs_used = self._select_chs(chs_used)
        self.n_chans = len(chs_used)
        self.w_pass_2d = w_pass_2d
        self.w_stop_2d = w_stop_2d

        self.fs_orig = fs_orig
        self.t_begin_buffer = t_begin_buffer
        self.chs_p3 = self._select_chs(chs_p3)
        self.w_pass_p3 = w_pass_p3
        self.w_stop_p3 = w_stop_p3
        self.fs_down_p3 = fs_down_p3
        self.t_begin_p3 = t_begin_p3
        self.t_end_p3 = t_end_p3

        self.results_predict = []
        self.symbol_predict = []
        self.loc_p3 = []
        self.loc_ssvep = []


    def _select_chs(self, chs_list):
        """Select Channels and Convert to channels index according to specified channels' name (e.g. Poz, Oz)

        Parameters
        ----------
        chs_list: list,
            channels' name list, e.g. ['POZ', 'Oz', 'po3']

        Returns
        -------
        idx_loc: list,
            index of selected channels, e.g. [22, 33, 35, 56]
        """
        idx_loc = list()
        if isinstance(chs_list, list):
            for char_value in chs_list:
                idx_loc.append(self._CHANNELS.index(char_value.upper()))

        return idx_loc

    def resample_data(self, raw_data, *args):
        """Down-sampling data from self.fs_orig to fs_down Hz.
        Default fs_down = self.fs_down. If want to use optional params, the first element of args should be fs_down.

        Parameters
        ----------
        raw_data: ndarray, 2-D. Generally, it's self.raw_data from the method data_from_buffer.
            axis 0: EEG channels.
            axis 1: the time points.
        *args: tuple. Only first element is valid. The other elements can also be developed to extend the functions.
            args[0]: int, down-sampling frequency, unit: Hz.
            args[1]: ndarray, events.

        Returns
        -------
        raw_data_resample: 2-D ndarray, the resampled data.
            axis 0: all EEG channels.
            axis 1: the time points.
        events: 2-D ndarray, all event values and latencies.
            n_events * 2(i.e. value and latency).
        """
        n_points = raw_data.shape[1]
        fs_down, evt = args[0], args[1]
        # fs_down, *_= args

        if self.fs_orig > fs_down:
            events = np.zeros_like(evt)
            # TODO resample better way
            raw_data_resample = signal.resample(raw_data, int(np.ceil(fs_down * n_points / self.fs_orig)), axis=-1)
            events[:, 0], events[:, -1] = evt[:, 0], (evt[:, -1]/(self.fs_orig/fs_down)).astype(int)
            # events[:, 0], events[:, -1] = evt[:, 0], np.round((evt[:, -1]/(self.fs_orig/fs_down))).astype(int)

            return raw_data_resample, events

        elif self.fs_orig == fs_down:
            # self.raw_data is raw_data_resample.
            return raw_data, evt

        else:
            raise ValueError('Oversampling is NOT recommended. The reason is self.fs < self.fs_down.')

    def filtered_data_iir_2(self, raw_data, *args):
        """Demo returned filtered_data is ndarray that is convenient for the following matrix manipulation.
        This way may be marginally faster in contrast to the dict type, due to supporting slices.

        Parameters
        ----------
        raw_data: ndarray, 2-D. It can be raw EEG data, or resampled data.
            axis 0: EEG channels.
            axis 1: the time points.
        *args: tuple, Only first three elements are valid. Passband and Stopband edge frequencies.
            args[0]: w_pass, 2-D ndarray. e.g. =np.array([[8, 18, 28, 38, 48, 58, 0.5], [72, 72, 72, 72, 72, 72, 10]])
            args[1]: w_stop, 2-D ndarray. e.g. =np.array([[6, 16, 26, 36, 46, 56, 0.1], [74, 74, 74, 74, 74, 74, 12]])
            args[2]: the sampling rate, unit: Hz. If None, default is self.fs_down.

        Returns
        -------
        filtered_data: ndarray of shape 3-D (n_chs, n_pnts, n_filters),
            axis 0: EEG channels.
            axis 1: the time points.
            axis 2: the different filters using filter bank method.
        """
        if len(args) == 0:
            w_pass, w_stop, fs_down = self.w_pass_2d, self.w_stop_2d, self.fs_down
        elif len(args) == 3:
            w_pass, w_stop, fs_down = args
        elif len(args) == 2:
            raise ValueError('The sampling rate(i.e. args[2]) corresponding to the band-pass band should be defined')
        else:
            raise ValueError('Expected two elements of args but %d were given.' % len(args))

        sos_system = dict()
        n_filters = w_pass.shape[1]
        n_chs, n_pnts = raw_data.shape
        filtered_data = np.empty((n_chs, n_pnts, n_filters))
        for idx_filter in range(n_filters):
            sos_system['filter'+str(idx_filter+1)] = \
                self._get_iir_sos_band([w_pass[0, idx_filter], w_pass[1, idx_filter]],
                                       [w_stop[0, idx_filter], w_stop[1, idx_filter]], fs_down)

            filtered_data[..., idx_filter] = \
                signal.sosfiltfilt(sos_system['filter'+str(idx_filter+1)], raw_data, axis=-1)

        return filtered_data

    def _get_iir_sos_band(self, w_pass, w_stop, *args):
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
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        fs_down = self.fs_down if len(args) == 0 else args[0]

        wp = [2 * w_pass[0] / fs_down, 2 * w_pass[1] / fs_down]
        ws = [2 * w_stop[0] / fs_down, 2 * w_stop[1] / fs_down]
        gpass = 4  # it's -3dB when setting as 3.
        gstop = 30  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

        return sos_system

    def fusion_coef(self, n_filters):
        weight_a = np.zeros(n_filters)
        for idx_filter in range(n_filters):
            idx_filter += 1
            weight_a[idx_filter - 1] = idx_filter ** (-1.25) + 0.25

        return weight_a

    def rr_coef_rnd(self, rr_coef):
        """Sum decision values along round axis.

        Parameters
        ----------
        rr_coef: decision value, 2-D ndarray: (**, 1)

        Returns
        -------
        rr_coef.sum(axis=0): 1-D ndarray.
        """
        rr_coef = rr_coef.reshape(self.n_rnds, 6)

        return rr_coef.sum(axis=0)

    def load_model(self):
        # load training model
        self.clf_p3 = joblib.load('clf_p3.joblib')
        self.clf_ssvep = joblib.load('clf_ssvep.joblib')
        print('**** ML Model loaded ****')

    def load_data(self, file_path: str, n_cnts: int):
        """Load data, Concatenate raw data, and Selected channels.

        Parameters
        ----------
        file_path: str
        n_cnts: int

        Returns
        -------
        data_all: ndarray of shape (n_eeg_chs+1, n_pnts)
                all EEG channels + label channel. Note the event is in the label channel.
        """

        raw_cnts = []
        for idx_cnt in range(1, n_cnts+1):
            file_name = os.path.join(file_path, str(idx_cnt)+'.cnt')

            # montage = mne.channels.make_standard_montage('standard_1020')
            raw_cnt = mne.io.read_raw_cnt(file_name, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True,
                                          verbose=False)

            # raw_cnt.filter(l_freq=0.1, h_freq=None, picks='eeg', n_jobs=4)  # remove slow drifts
            # raw_cnt.filter(l_freq=None, h_freq=90, picks='eeg', n_jobs=4)  # 1/3 sampling rate
            raw_cnts.append(raw_cnt)
        raw_cnts_mne = concatenate_raws(raw_cnts)

        # custom mapping for event id
        custom_mapping_p3 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '251': 251}
        custom_mapping_ssvep = {'4': 4}
        for idx_command in range(1, self.num_commands+1):
            custom_mapping_p3[str(idx_command+12)] = idx_command + 12
            custom_mapping_ssvep[str(idx_command+12)] = idx_command + 12

        events, events_ids = mne.events_from_annotations(raw_cnts_mne, event_id=custom_mapping_p3)

        data_tmp = raw_cnts_mne.get_data()
        evt_tmp = np.zeros((1, data_tmp.shape[-1]))
        evt_tmp[0, events[:, 0]] = events[:, -1]

        data_all = np.vstack((data_tmp, evt_tmp))

        self.label_tars = events[:, -1].reshape((-1, 2 + self.n_rnds * 6))[:, 0] - 12

        return data_all

    def ext_block(self, data_all):
        """Sort type and latency.
        data_all:  2-D, numpy
            all EEG channels + label channel. Note the event is in the label channel.
        data_online: 3-D, ndarray
            (all EEG channels + label channel) * n_pnts * n_test_blocks
        """

        events = data_all[-1, :]
        loc_ending = np.argwhere(events == 251).squeeze()
        loc_begin = loc_ending - int(np.round(self.fs_orig * self.t_begin_buffer)) + 1
        self.n_blocks, n_pnts = len(loc_ending), int(loc_ending[0]-loc_begin[0])+1

        data_online = np.empty((data_all.shape[0], n_pnts, self.n_blocks))
        for idx_block in range(self.n_blocks):
            data_online[..., idx_block] = data_all[:, loc_begin[idx_block]:loc_ending[idx_block]+1]

        return data_online

    def recognition(self, data_online):
        """Recognize features and Output accuracy.

        Parameters
        ----------
        data_online: 3-D, ndarray
            (all EEG channels + label channel) * n_pnts * n_test_blocks
        """
        raw_data_all, events_tmp = data_online[:-1, ...] * 10e6, data_online[-1, ...].astype(int)
        evt_latency = np.array([np.argwhere(events_tmp[:, idx_block] != 0).squeeze() for idx_block in range(self.n_blocks)])
        evt_value = np.array([events_tmp[evt_latency[idx_block], idx_block] for idx_block in range(self.n_blocks)])

        for idx_block in range(self.n_blocks):
            raw_data = raw_data_all[..., idx_block]
            events_bk = np.vstack((evt_value[idx_block, :], evt_latency[idx_block, :])).T

            # --------------------------------SSVEP processing---------------------------------------#
            data_tmp_ssvep, events_ssvep = self.resample_data(raw_data[self.chs_used, :], self.fs_down, events_bk)

            n_evts = events_ssvep.shape[0] - 2  # exclude the first and the last trigger
            if n_evts != (self.n_rnds * 6):
                raise ValueError('Some problems occurred. The trigger may be missed!')

            filtered_ssvep = self.filtered_data_iir_2(data_tmp_ssvep)

            n_filters = self.w_pass_2d.shape[-1]
            latency_1st = events_ssvep[1:-1:6, -1]  # the first little trigger latency

            latency_begin = int(np.ceil(self.fs_down * self.t_begin_cls)) + latency_1st - 1
            latency_end = int(np.ceil(self.fs_down * self.t_end_cls)) + latency_1st

            test_ssvep = np.empty((self.n_rnds, len(self.chs_used), latency_end[0] - latency_begin[0], n_filters))

            for idx_rnd in range(self.n_rnds):
                test_ssvep[idx_rnd, ...] = filtered_ssvep[:, latency_begin[idx_rnd]:latency_end[idx_rnd], :]

            rr_coef = np.zeros((n_filters, len(self.clf_ssvep[0].classes_), self.n_rnds))
            for idx_filter in range(n_filters):
                rr_coef[idx_filter, :] = self.clf_ssvep[idx_filter].transform(test_ssvep[..., idx_filter]).T

            if n_filters > 1:
                rr_coef = rr_coef ** 2
                weight_a = self.fusion_coef(n_filters)
                for idx_filter in range(n_filters):
                    rr_coef[idx_filter, :] *= weight_a[idx_filter]

            rr_coef = rr_coef.sum(axis=0, keepdims=False).sum(axis=-1)
            self.loc_ssvep.append(rr_coef.argmax(axis=0) + 1)

            # --------------------------------P300 processing-----------------------------------------#
            # data_refer = raw_data[self.chs_p3, :] - raw_data[32, :]
            data_tmp_p3, events_p3 = self.resample_data(raw_data[self.chs_p3, :], self.fs_down_p3, events_bk)
            # data_tmp_p3 -= signal.resample(raw_data[32, :], int(np.ceil(self.fs_down_p3 * raw_data.shape[-1] / 1000)))

            filtered_p3 = self.filtered_data_iir_2(data_tmp_p3, self.w_pass_p3, self.w_stop_p3,
                                                   self.fs_down_p3).squeeze()
            # n_filters_p3 = self.w_pass_p3.shape[-1]

            type_tmp, latency_tmp = events_p3[1:-1, 0], events_p3[1:-1, -1]
            type_all, latency_all = np.zeros_like(type_tmp, dtype=int), np.zeros_like(latency_tmp, dtype=int)
            for idx_evt in range(n_evts):
                type_value = type_tmp[idx_evt]
                # type_all[type_value-1] = type_value
                latency_all[(type_value - 1) + idx_evt // 6 *6] = latency_tmp[idx_evt]

            latency_begin = int(np.ceil(self.fs_down * self.t_begin_p3)) + latency_all - 1
            latency_end = int(np.ceil(self.fs_down * self.t_end_p3)) + latency_all
            n_pnts_p3 = self.clf_p3.coef_.shape[-1]

            test_p3 = np.empty((n_evts, n_pnts_p3))
            for idx_evt in range(n_evts):
                test_p3[idx_evt, :] = filtered_p3[:, latency_begin[idx_evt]:latency_end[idx_evt]:10].reshape((1, -1),
                                                                                                             order='C')
            # (6, )
            dv_p3 = self.clf_p3.transform(test_p3).squeeze() if self.n_rnds <= 1 else self.rr_coef_rnd(self.clf_p3.transform(test_p3))
            self.loc_p3.append(dv_p3.argmax(axis=-1) + 1)

            init_values = np.hstack((np.arange(1, 19), np.arange(109, 127)))
            loc_char = init_values[self.loc_ssvep[-1]-1] + (self.loc_p3[-1] - 1) * 18

            init_char_index = np.hstack((np.arange(0, 18), np.arange(108, 126)))

            self.results_predict.append(loc_char)
            self.symbol_predict.append(symbol_216[init_char_index[self.loc_ssvep[-1]-1] + (self.loc_p3[-1] - 1) * 18])

        label_ssvep, label_p3 = [], []
        for idx_tar in range(len(self.label_tars)):
            tar_tmp = self.label_tars[idx_tar] - 1
            tar_tmp2 = self.label_tars[idx_tar]
            if tar_tmp <= 107:
                label_p3.append(tar_tmp // 18 + 1)
                label_ssvep.append(tar_tmp2 % 18)
            else:
                label_p3.append(tar_tmp // 18 - 6 + 1)
                label_ssvep.append(tar_tmp2 % 18 + 18)

        label_ssvep = np.array(label_ssvep)
        label_p3 = np.array(label_p3)
        label_ssvep[np.argwhere(label_ssvep == 18)] = 36
        label_ssvep[np.argwhere(label_ssvep == 0)] = 18

        acc_p3 = (label_p3 == np.array(self.loc_p3)).sum() / len(label_p3)
        acc_ssvep = (label_ssvep == np.array(self.loc_ssvep)).sum() / len(label_p3)

        acc = (self.label_tars == np.array(self.results_predict)).sum() / self.n_blocks
        print('The simulated online accuracy is %.4f' % acc)
        print('bk')


if __name__ == '__main__':
    begin_time = time.time()
    file_path = r'.\samples\on'

    tmin, tmax = -0.2, 1.4
    fs_down = 250
    time_ssvep = 0.7
    n_cnts = 1
    n_rounds = 5
    t_cut = 5.1  # if n_rounds == 1 else 3.7
    # [2.3, 3, 3.7, 4.4, 5.1, 5.8]

    # Initialization class ProcessRecog
    w_pass_2d = np.array([[8, 18, 28, 38, 48, 58], [72, 72, 72, 72, 72, 72]])  # 70
    w_stop_2d = np.array([[6, 16, 26, 36, 46, 56], [74, 74, 74, 74, 74, 74]])  # 72
    w_pass_p3 = np.array([[0.5], [10]])
    w_stop_p3 = np.array([[0.1], [12]])

    # p3: ['FCZ', 'CZ', 'pz', 'po7', 'po8', 'oz']
    # ssvep: ['POZ','PZ','PO3','PO5','PO4','PO6','O1','OZ','O2']
    process_recog = ProcessingRecog(n_rounds=n_rounds, t_begin_cls=0.14, t_end_cls=0.14+0.8, fs_down=250, CHANNELS=CHANNELS,
                               chs_used=['cpz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'PZ', 'P1',
                                         'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3', 'PO4', 'PO5', 'PO6',
                                         'PO7', 'PO8', 'OZ', 'O1', 'O2', 'CB1', 'CB2'],
                               w_pass_2d=w_pass_2d, w_stop_2d=w_stop_2d, fs_orig=1000,t_begin_buffer=t_cut,
                               chs_p3=['F3', 'Fz', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO7', 'PO8', 'OZ'],
                               w_pass_p3=w_pass_p3, w_stop_p3=w_stop_p3, fs_down_p3=250, t_begin_p3=0.05, t_end_p3=0.8,
                               )
    process_recog.load_model()
    data_all = process_recog.load_data(file_path, n_cnts=n_cnts)

    data_online = process_recog.ext_block(data_all)
    process_recog.recognition(data_online)
    print(time.time() - begin_time)

    print('breakpoint')

