# -*- coding:utf-8 -*-
'''
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2022-03-14
update: 2022-06
version 1.0

Application:
    Build Offline model for training problems.
    Demo is developed based on the hybrid P300-SSVEP BCI.
    When the manuscript is published online, the corresponding sample data for this .py will be uploaded to github.

'''
import os
import warnings
import time

import numpy as np
import scipy.io as sio
from scipy import signal
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
# import h5py
import mne
from mne.io import concatenate_raws
from mne import Epochs
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from BasePreProcessing import BasePreProcessing
from algorithms.tdca import TDCA, generate_cca_references


warnings.filterwarnings('ignore')  # or warnings.filterwarnings('ignore')

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

def select_chans(chans_list):
    """Select Channels and Convert to channels index according to specified channels' name (e.g. Poz, Oz)

    Parameters
    ----------
    chans_list: list,
        channels' name list, e.g. ['POZ', 'Oz', 'po3']

    Returns
    -------
    idx_loc: list,
        index of selected channels, e.g. [22, 33, 35, 56]
    """
    idx_loc = list()
    if isinstance(chans_list, list):
        for char_value in chans_list:
            idx_loc.append(CHANNELS.index(char_value.upper()))

    return idx_loc


class PreProcessing(BasePreProcessing):

    num_commands = 216  # 54 commands
    def __init__(self, filepath, t_begin, t_end, n_classes=36, n_rounds=6, n_tdma=6, tmin=-0.4, tmax=1.4, fs_down=250, chans=None):
        super().__init__(filepath, t_begin, t_end, fs_down, chans, n_classes=n_classes)
        self.n_rnds = n_rounds
        self.n_tdma = n_tdma
        self.tmin = tmin
        self.tmax = tmax

    def select_chans(self, chans_list):

        idx_loc = list()
        if isinstance(chans_list, list):
            for char_value in chans_list:
                idx_loc.append(self.CHANNELS.index(char_value.upper()))

        return idx_loc

    def load_data(self):
        """Load data, Concatenate raw data, and Selected channels.

        Returns
        -------
        data_epoch_p3: object of mne.Epoch.
        data_epoch_ssvep: object of mne.Epoch.
        """

        raw_cnts = []
        for idx_cnt in range(1, 7):
            file_name = os.path.join(self.filepath, str(idx_cnt)+'.cnt')

            # montage = mne.channels.make_standard_montage('standard_1020')
            raw_cnt = mne.io.read_raw_cnt(file_name, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True,
                                          verbose=False)

            # raw_cnt.filter(l_freq=0.1, h_freq=None, picks='eeg', n_jobs=4)  # remove slow drifts
            # raw_cnt.filter(l_freq=None, h_freq=90, picks='eeg', n_jobs=4)  # 1/3 sampling rate
            raw_cnts.append(raw_cnt)
        raw_cnts_mne = concatenate_raws(raw_cnts)

        # custom mapping for event id
        custom_mapping_p3 = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        custom_mapping_ssvep = {'4': 4}
        for idx_command in range(1, self.num_commands+1):
            custom_mapping_p3[str(idx_command+12)] = idx_command + 12
            custom_mapping_ssvep[str(idx_command+12)] = idx_command + 12

        events_p3, events_ids_p3 = mne.events_from_annotations(raw_cnts_mne, event_id=custom_mapping_p3)
        events_ssvep, events_ids_ssvep = mne.events_from_annotations(raw_cnts_mne, event_id=custom_mapping_ssvep)

        # drop_chans = ['M1', 'M2']  # drop channels

        # picks = mne.pick_types(raw_cnts_mne.info, emg=False, eeg=True, stim=False, eog=False,
        #                        exclude=drop_chans, selection=['POZ','PZ','PO3','PO5','PO4','PO6','O1','OZ','O2'])
        picks = mne.pick_types(raw_cnts_mne.info, emg=False, eeg=True, stim=False, eog=False,
                               selection=None)  # exclude=drop_chans,
        picks_ch_names = [raw_cnts_mne.ch_names[i] for i in picks]  # layout picked channels' name
        
        # raw_data = raw_cnts_mne.pick(picks)  # specify channels

        # baseline correction+down-sampling(self.tmin, 0)
        data_epoch_p3 = mne.Epochs(raw_cnts_mne, events=events_p3, event_id=events_ids_p3, tmin=self.tmin, picks=picks,
                                   tmax=self.tmax, baseline=None, detrend=0, decim=int(1000/self.fs_down), preload=True)

        data_epoch_ssvep = mne.Epochs(raw_cnts_mne, events=events_ssvep, event_id=events_ids_ssvep, tmin=self.tmin, picks=picks,
                                      tmax=self.tmax, baseline=None, detrend=0, decim=int(1000/self.fs_down), preload=True)

        return data_epoch_p3, data_epoch_ssvep

    def ext_label(self, events_all):
        """Sort type and latency.

        Parameters
        ----------
        events_all: ndarray of shape 2-D.
            events_all[0, :]: latency
            events_all[1, :]: label

        Returns
        -------
        tar_class_ssvep, label_p3, loc_origin
        """
        _, num_events = events_all.shape
        type_all, latency_all = events_all[1], events_all[0]
        types = type_all.reshape((-1, 37), order='C')  # 36 freqs, 6 targets, 6 rounds
        latencies = latency_all.reshape((-1, 37), order='C')

        target_type = types[:, 0] - 12  # big label
        n_tars = len(target_type)
        tar_class_ssvep = np.zeros_like(target_type)
        for idx_class in range(n_tars):
            tmp_value = target_type[idx_class]
            if  tmp_value <= 108:
                tar_class_ssvep[idx_class] = 18 if np.mod(tmp_value, 18)==0 else np.mod(tmp_value, 18)
            else:
                tar_class_ssvep[idx_class] = 36 if np.mod(tmp_value, 18) == 0 else np.mod(tmp_value, 18)+18

        types, latencies = np.delete(types, 0, axis=1), np.delete(latencies, 0, axis=1)
        types, latencies = types.T, latencies.T

        # latency_sort = np.zeros_like(latencies)
        # type_sort = np.zeros_like(types)
        loc_origin = np.zeros_like(types)
        for idx_tar in range(n_tars):
            for idx_round in range(self.n_rnds):
                for idx_char in range(self.n_tdma):
                    tmp = types[idx_char+idx_round*self.n_tdma, idx_tar]
                    # latency_sort[tmp+idx_round*self.n_tdma-1, idx_tar] = latencies[idx_char+idx_round*self.n_tdma, idx_tar]
                    # type_sort[tmp+idx_round*self.n_tdma-1, idx_tar] = types[idx_char+idx_round*self.n_tdma, idx_tar]
                    loc_origin[tmp+idx_round*self.n_tdma-1, idx_tar] = idx_char + idx_round * self.n_tdma + 1 + idx_tar * 37

        loc_tmp = np.arange(0, n_tars*37, 37)

        # used to sort data sequence corresponding to sorted trial
        loc_origin = np.concatenate((np.expand_dims(loc_tmp, axis=0), loc_origin), axis=0).T  # (targets*trials)
        # 1-D vector
        # loc_origin = np.concatenate((np.expand_dims(loc_tmp, axis=0), loc_origin), axis=0).reshape((-1,), order='F')

        # target_type, type_sort, latency_sort are for cut data to get Epoch, though no used here.

        # for mVEP label
        label_p3 = np.zeros_like(types) == 1
        label_tmp = np.zeros_like(target_type)
        for idx_p3 in range(n_tars):
            tmp_value = target_type[idx_p3] - 1
            if tmp_value <= 107:
                label_tmp[idx_p3] = tmp_value // 18
            else:
                label_tmp[idx_p3] = tmp_value // 18 - 6

        label_buff = np.zeros((self.n_rnds, n_tars), dtype=int)
        label_buff[0, :] = label_tmp
        for idx_rnd in range(1, self.n_rnds):
            label_buff[idx_rnd, :] = label_tmp + idx_rnd * self.n_tdma

        for idx_tar in range(n_tars):
            label_p3[label_buff[:, idx_tar], idx_tar] = True

        return tar_class_ssvep, label_p3, loc_origin

    def ext_epochs(self, raw_data, events_all):
        pass

    def resample_data(self, raw_data):
        pass

    def filtered_data_iir(self, w_pass_2d, w_stop_2d, data):
        """Filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.

        Parameters
        ----------
        w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
            e.g.
            w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54], [70, 70, 70, 70, 70, 70, 70]])
            w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52], [72, 72, 72, 72, 72, 72, 72]])
        data: 4-d, numpy, from method load_data or resample_data.
            n_targets * n_trials * n_chans * n_samples

        Returns
        -------
        filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_targets * n_trials * n_chans * n_samples.
        """

        self.n_filter = w_stop_2d.shape[1]
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.n_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.n_filter):
            sos_system['filter' + str(idx_filter + 1)] = self._get_iir_sos_band(
                w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                w_stop=[w_stop_2d[0, idx_filter],
                        w_stop_2d[1, idx_filter]])
            filtered_data['bank' + str(idx_filter + 1)] = \
                signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=-1)

        return filtered_data


class Classification:
    """
    Parameters
    ----------
    n_splits: int,
        the fold of cross validation.
        if None or equal to n_used_trials, Leave-one-out will be used in subsequent processing.
    n_classes: int,
        the number of classes.
    n_rounds: int,
        the number of rounds.
    n_tdma: int
    tmin, tmax: float, unit: second.
        the maximum and minimum value of the time window for cutting epoch.
    fs_down: int,
        down-sampling rate.
    t_ssvep_min, t_ssvep_max: float, unit: second.
        the using time window of SSVEP recognition.
    t_p3_min, t_p3_max: float, unit: second.
        the using time window of P300 recognition
    l_delay: int,
        the delay of tdca.
    n_components_tdca: int,
        the number of components for tdca.
    chs_ssvep, chs_p3: list | None
        the channels for SSVEP or P300.
    """

    def __init__(self, n_splits=None, n_classes=36, n_rounds=6, n_tdma=6, tmin=-0.2, tmax=1.2, fs_down=250,
                 t_ssvep_min=0.14, t_ssvep_max=1., t_p3_min=0., t_p3_max=0.8, l_delay=5, n_components_tdca=1,
                 chs_ssvep=None, chs_p3=None):

        self.n_splits = n_splits  # default Leave-one-out
        self.n_classes = n_classes
        self.n_rounds = n_rounds
        self.n_tdma = n_tdma
        self.tmin = tmin
        self.tmax = tmax
        self.fs_down = fs_down
        self.t_ssvep_min = t_ssvep_min
        self.t_ssvep_max = t_ssvep_max
        self.t_p3_min = t_p3_min
        self.t_p3_max = t_p3_max
        self.l_delay = l_delay
        self.n_components = n_components_tdca
        self.chs_ssvep = select_chans(chs_ssvep)
        self.chs_p3 = select_chans(chs_p3)


    def selected_trials(self, all_data, all_label):
        """Select trials from all trials according a fixed random sequence.

        Parameters
        ----------
        all_data: dict,
            {'bank1': values1, 'bank2': values2, ..., 'bank'+str(num_filter): values}
            values1, values2, ...: n_trials * n_chans * n_samples
        all_label: 2-d numpy,
            all_label.shape: (n_trials, )

        Returns
        -------
        all_data, all_label.
            Same data format with input variables.
        """
        self.n_filter_ssvep = len(list(all_data.keys())) - 1
        self.n_tars = len(all_label)
        train_set, test_set = dict(), dict()

        for idx_filter in range(self.n_filter_ssvep):
            idx_filter += 1
            data_temp = all_data['bank'+str(idx_filter)]
            data_train, data_test, loc_tar_train, loc_tar_test = train_test_split(data_temp, np.arange(0, self.n_tars),
                                                                   train_size=(self.n_splits-1)*self.n_classes,
                                                                   random_state=123, stratify=all_label)
            train_set['bank'+str(idx_filter)] = data_train
            test_set['bank'+str(idx_filter)] = data_test

        label_train = all_label[loc_tar_train]
        label_test = all_label[loc_tar_test]

        return train_set, test_set, loc_tar_train, loc_tar_test, label_train, label_test

    def fusion_coef(self):
        weight_a = np.zeros(self.n_filter_ssvep)
        for idx_filter in range(self.n_filter_ssvep):
            idx_filter += 1
            weight_a[idx_filter-1] = idx_filter**(-1.25)+0.25

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

        rr_coef_rnd = np.zeros_like(rr_coef)
        rr_coef_rnd[:, 0, :] = rr_coef[:, 0, :]
        for idx_rnd in range(1, self.n_rounds):
            rr_coef_rnd[:, idx_rnd, :] = rr_coef[:, :(idx_rnd+1), :].sum(axis=1, keepdims=False)

        return rr_coef_rnd

    def recognition(self, filtered_data_p3, filtered_data_ssvep, tar_class_ssvep, label_p3, Yf):
        '''Build Model.

        Parameters
        ----------
        filtered_data_p3: dict,
            {'bank1': values1, 'bank2': values2, ..., 'bank'+str(num_filter): values}
            values1, values2, ...: n_trials * n_chans * n_samples
        filtered_data_ssvep: dict,
            {'bank1': values1, 'bank2': values2, ..., 'bank'+str(num_filter): values}
            values1, values2, ...: n_trials * n_chans * n_samples
        all_label: ndarray of shape (self.n_tars, ).

        Returns
        -------
        clf_p3.joblib: classification model for P300-based BCI.
        clf_ssvep.joblib: classification model for SSVEP-based BCI.
        '''
        self.n_filter_ssvep = len(list(filtered_data_ssvep.keys()))
        self.n_tars = len(tar_class_ssvep)

        #---------------------Parameters settings--------------------------------------#
        begin_pnt_ssvep = int(np.ceil((self.t_ssvep_min - self.tmin) * self.fs_down)) - 1
        end_pnt_ssvep = int(np.ceil((self.t_ssvep_max - self.tmin) * self.fs_down))

        begin_pnt_p3 = int(np.ceil((self.t_p3_min - self.tmin) * self.fs_down)) - 1
        end_pnt_p3 = int(np.ceil((self.t_p3_max - self.tmin) * self.fs_down))
        # begin_pnt_p3 = int((self.t_p3_min - self.tmin) * self.fs_down)
        # end_pnt_p3 = int((self.t_p3_max - self.tmin) * self.fs_down) + 1

        _, n_trials, _, n_samples = filtered_data_p3['bank1'].shape

        # -----------------------------------------P300 Model----------------------------------------#
        data_tmp_p3 = np.delete(filtered_data_p3['bank1'], 0, axis=1)  # n_tars * n_trials * n_chans * n_times
        # tars
        data_tmp = data_tmp_p3[:, :, self.chs_p3, begin_pnt_p3:end_pnt_p3:10].reshape((self.n_tars, n_trials - 1, -1), order='C')
        tars_data = data_tmp[label_p3.T, :].T.reshape((-1, self.n_rounds, self.n_tars)).transpose((1, 2, 0))
        n_feats_p3 = tars_data.shape[-1]
        p3_tars_data = tars_data.reshape((-1, n_feats_p3), order='F')  # samples*feats

        # non-tars
        nontars_data = data_tmp[~label_p3.T, :].T.reshape((-1, self.n_rounds*(self.n_tdma-1), self.n_tars)).transpose((1, 2, 0))  # [[1, 8, 12, 16, 20, 29], :, :]
        p3_non_tars_data = nontars_data.reshape((-1, n_feats_p3), order='F')  # samples*feats

        n_lda_samples = p3_tars_data.shape[0]
        y_lda = np.hstack((np.ones(n_lda_samples, dtype=int), np.ones(n_lda_samples*5, dtype=int) * -1))

        clf_p3 = LDA()  # solver='eigen', shrinkage=0.1
        clf_p3.fit(np.vstack((p3_tars_data, p3_non_tars_data)), y_lda)

        # way-1: Recommended
        import joblib
        joblib.dump(clf_p3, 'clf_p3.joblib')
        # clf_p3 = joblib.load('clf_p3.joblib')

        # way-2: Not Recommended
        # np.savez('model_p3', p3_xbar=clf_p3.xbar_, p3_scalings=clf_p3.scalings_, p3_max_components=clf_p3._max_components)

        # -----------------------------------------SSVEP Recognition----------------------------------------#
        clf_ssvep = []
        for idx_filter in range(self.n_filter_ssvep):

            data_temp = filtered_data_ssvep['bank'+str(idx_filter+1)]  # n_tars * n_trials * n_chans * n_times

            train_set = data_temp[:, :, self.chs_ssvep, begin_pnt_ssvep:end_pnt_ssvep].mean(axis=1)

            clf = TDCA(l=self.l_delay, n_components=self.n_components)
            clf.fit(train_set, tar_class_ssvep, Yf)
            clf_ssvep.append(clf)
            # n_test_samples * rounds * 36 coefs
            # feat = clf.transform(test_set_tdca).reshape((-1, self.n_rounds, self.n_classes)).transpose((2, 1, 0))
            # rr_coef[idx_filter, ...] = feat

        import joblib
        joblib.dump(clf_ssvep, 'clf_ssvep.joblib')

        print('Congratulation! Model has been saved at the same directory.')
        print('**** Model name: clf_p3.joblib, clf_ssvep.joblib. ****')


if __name__ == '__main__':
    begin_time = time.time()
    n_classes, n_tdma, n_rounds = 36, 6, 6
    tmin, tmax = -0.2, 1.4
    fs_down = 250
    time_ssvep = 0.7

    file_path = r'.\samples'

    # Generate Yf
    stim_freq = np.arange(10.4, 17.6, 0.2)
    Yf = [generate_cca_references(stim_freq[idx_freq], fs_down, time_ssvep, n_harmonics=5) for idx_freq in range(n_classes)]
    Yf = np.concatenate(Yf)  # 36freqs * 10 sin-cos * n_pnts

    # p3: ['FCZ', 'CZ', 'pz', 'po7', 'po8', 'oz']
    # ssvep: ['POZ','PZ','PO3','PO5','PO4','PO6','O1','OZ','O2']
    preEEG =  PreProcessing(file_path, t_begin=0.05, t_end=0.8, n_classes=n_classes, n_rounds=n_rounds, n_tdma=n_tdma,
                            tmin=tmin, tmax=tmax,
                            fs_down=fs_down, chans=None,
                            )

    # raw_data, events = preEEG.load_data()
    data_epoch_p3, data_epoch_ssvep = preEEG.load_data()

    # data_epoch_p3, _ = mne.set_eeg_reference(data_epoch_p3, ref_channels=['M1'])
    # data_epoch_ssvep, _ = mne.set_eeg_reference(data_epoch_ssvep, ref_channels=['M1'])

    events_p3 = np.delete(data_epoch_p3.events, 1, axis=1).transpose((1, 0))
    events_ssvep = np.delete(data_epoch_ssvep.events, 1, axis=1).transpose((1, 0))
    loc_ssvep = np.delete(np.arange(0, 216*7).reshape((-1, 7)), 0, axis=1)

    tar_class_ssvep, label_p3, loc_origin = preEEG.ext_label(events_p3)
    n_tars = len(tar_class_ssvep)
    # data = data_epoch.get_data()
    # n_targets * n_trials * n_chans * n_samples, (or n_chans * n_samples * n_trials * n_targets)
    all_data_p3 = data_epoch_p3.get_data()[loc_origin, :, :] * 1e6  # .transpose((2, 3, 1, 0))
    all_data_ssvep = data_epoch_ssvep.get_data()[loc_ssvep, :, :] * 1e6  # .transpose((2, 3, 1, 0))

    w_pass_p3 = np.array([[0.5], [10]])  # 70
    w_stop_p3 = np.array([[0.1], [12]])  # 72
    filtered_data_p3 = preEEG.filtered_data_iir(w_pass_p3, w_stop_p3, all_data_p3)

    w_pass_2d = np.array([[8, 18, 28, 38, 48, 58], [72, 72, 72, 72, 72, 72]])  # 70
    w_stop_2d = np.array([[6, 16, 26, 36, 46, 56], [74, 74, 74, 74, 74, 74]])  # 72
    filtered_data_ssvep = preEEG.filtered_data_iir(w_pass_2d, w_stop_2d, all_data_ssvep)

    del all_data_ssvep, all_data_p3, data_epoch_p3, data_epoch_ssvep

    n_splits = int(n_tars/n_classes)

    # classical 9 channels: ['POZ','PZ','PO3','PO5','PO6','PO4','O1','OZ','O2','PO7','PO8']
    # classical P3 channels: ['Fz', 'Cz', 'Pz', 'PO7', 'PO8', 'Oz']
    # 16 channels-STDA using: ['F3', 'Fz', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO7', 'PO8', 'OZ']
    classifyEEG = Classification(n_splits=n_splits, n_classes=n_classes, n_rounds=n_rounds, n_tdma=n_tdma,
                                 tmin=tmin, tmax=tmax, fs_down=preEEG.fs_down,
                                 t_ssvep_min=0.14, t_ssvep_max=0.14+0.8, t_p3_min=0.05, t_p3_max=0.8,
                                 l_delay=3, n_components_tdca=1,
                                 chs_ssvep=['cpz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'PZ',
                                            'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3', 'PO4', 'PO5',
                                            'PO6', 'PO7', 'PO8', 'OZ', 'O1', 'O2', 'CB1', 'CB2'],
                                 chs_p3=['F3', 'Fz', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO7', 'PO8', 'OZ'])
    # train_set, test_set, loc_tar_train, loc_tar_test, label_train, label_test = classifyEEG.selected_trials(filtered_data, tar_class_ssvep)

    acc = classifyEEG.recognition(filtered_data_p3, filtered_data_ssvep, tar_class_ssvep, label_p3, Yf)

    print(time.time() - begin_time)

    print('breakpoint')

