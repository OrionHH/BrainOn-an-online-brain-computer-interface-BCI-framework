# -*- coding:utf-8 -*-
'''
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2020-07-18
update: 2022-06
version 1.0

Application: Demo of BCI Online modulation and processing base framework.

'''
import time, sys
sys.path.append('..')
import threading
import warnings

import numpy as np
# from scipy import signal
import joblib

from ReadNeuroscan import ReadNeuroscan
from BaseFramework import BaseProcessingRecog
from BaseFramework import SendMessageUdp


warnings.filterwarnings('ignore')  # or warnings.filterwarnings("default")


class ReadData(ReadNeuroscan):

    # also can override CHANNELS

    def __init__(self, fs_orig=1000, ip_address=None, dur_one_packet=0.04, time_buffer=30, end_flag_trial=251):
        super().__init__(fs_orig, ip_address, dur_one_packet, time_buffer, end_flag_trial)


class ProcessingRecog(BaseProcessingRecog):

    def __init__(self, t_begin_cls, t_end_cls, fs_down, CHANNELS, chs_used, w_pass_2d, w_stop_2d, event_thread_process,
                 n_points_buffer, fs_orig, end_flag_trial, t_begin_buffer, chs_p3, w_pass_p3=None, w_stop_p3=None,
                 fs_down_p3=None, t_begin_p3=0.05, t_end_p3=0.8, n_rounds=1, flag_process=False):
        super().__init__(t_begin_cls, t_end_cls, fs_down, CHANNELS, chs_used, w_pass_2d, w_stop_2d, event_thread_process,
                         n_points_buffer, fs_orig, end_flag_trial, t_begin_buffer, t_end_buffer=0, raw_data=None)
        self.flag_process = flag_process
        self.chs_p3 = self._select_chs(chs_p3)
        self.w_pass_p3 = w_pass_p3
        self.w_stop_p3 = w_stop_p3
        self.fs_down_p3 = fs_down_p3
        self.t_begin_p3 = t_begin_p3
        self.t_end_p3 = t_end_p3
        self.n_rounds = n_rounds
        self.loc_p3 = []
        self.loc_ssvep = []

    def ext_epochs(self, filtered_data):
        """Cut epochs.
        Parameters
        ----------
        filtered_data: 3-D, numpy, n_chans * n_points * n_filters.

        self.evt: 2-D ndarray, from method channel_selected of base class BaseProcessingRecog.
                format - n_events * 2(i.e. value and latency)

        Returns
        -------
        new_epoch: ndarray of shape 3-D (n_chans * n_points * n_filters), e.g. (9, 126, 7)
        """
        if self.evt.shape[0] != 2:
            raise ValueError('The row of self.evt should be equal to 1.')

        t_trigger = self.evt[0, 1]
        latency_begin = int(round(self.fs_down * self.t_begin_cls) + t_trigger)
        latency_end = int(round(self.fs_down * self.t_end_cls) + t_trigger)

        new_epoch = filtered_data[:, latency_begin:latency_end + 1, :]

        if new_epoch.shape[1] == 125:
            print('problems')

        return new_epoch

    def fusion_coef(self, n_filters):
        weight_a = np.zeros(n_filters)
        for idx_filter in range(n_filters):
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
        rr_coef = rr_coef.reshape(self.n_rounds, 6)

        return rr_coef.sum(axis=0)

    def load_model(self):
        # load training model
        self.clf_p3 = joblib.load('../clf_p3.joblib')
        self.clf_ssvep = joblib.load('../clf_ssvep.joblib')
        print('**** ML Model loaded ****')

    def recognition(self):
        # --------------------------------SSVEP processing---------------------------------------#
        data_tmp_ssvep, events_ssvep = self.resample_data(self.raw_data[self.chs_used, :])
        n_evts = events_ssvep.shape[0] - 2  # exclude the first and the last trigger

        if n_evts != (self.n_rounds * 6):
            raise ValueError('Some problems occurred. The trigger may be missed!')

        filtered_ssvep = self.filtered_data_iir_2(data_tmp_ssvep)
        n_filters = self.w_pass_2d.shape[-1]
        latency_1st = events_ssvep[1:-1:6, -1]  # the first little trigger latency

        latency_begin = int(np.ceil(self.fs_down * self.t_begin_cls)) + latency_1st - 1
        latency_end = int(np.ceil(self.fs_down * self.t_end_cls)) + latency_1st

        test_ssvep = np.empty((self.n_rounds, len(self.chs_used), latency_end[0]-latency_begin[0], n_filters))

        for idx_rnd in range(self.n_rounds):
            test_ssvep[idx_rnd, ...] = filtered_ssvep[:, latency_begin[idx_rnd]:latency_end[idx_rnd], :]

        rr_coef = np.zeros((n_filters, len(self.clf_ssvep[0].classes_), self.n_rounds))
        for idx_filter in range(n_filters):
            rr_coef[idx_filter, :] = self.clf_ssvep[idx_filter].transform(test_ssvep[..., idx_filter]).T

        if n_filters > 1:
            rr_coef = rr_coef ** 2
            weight_a = self.fusion_coef(n_filters)
            for idx_filter in range(n_filters):
                rr_coef[idx_filter, :] *= weight_a[idx_filter]

        rr_coef = rr_coef.sum(axis=0, keepdims=False).sum(axis=-1)
        self.loc_ssvep.append(rr_coef.argmax(axis=0) + 1)
        # loc_ssvep = rr_coef.argmax(axis=0)

        # --------------------------------P300 processing-----------------------------------------#
        data_refer = self.raw_data[self.chs_p3, :] - self.raw_data[32, :]
        data_tmp_p3, events_p3 = self.resample_data(data_refer, self.fs_down_p3)
        # data_tmp_p3, events_p3 = self.resample_data(self.raw_data[self.chs_p3, :], self.fs_down_p3)
        # data_tmp_p3 -= signal.resample(self.raw_data[32, :], int(np.ceil(self.fs_down_p3 * self.raw_data.shape[-1] / 1000)))

        filtered_p3 = self.filtered_data_iir_2(data_tmp_p3, self.w_pass_p3, self.w_stop_p3, self.fs_down_p3).squeeze()
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
            test_p3[idx_evt, :] = filtered_p3[:, latency_begin[idx_evt]:latency_end[idx_evt]:10].reshape((1, -1), order='C')

        # (6, )
        dv_p3 = self.clf_p3.transform(test_p3).squeeze() if self.n_rounds <= 1 else self.rr_coef_rnd(self.clf_p3.transform(test_p3))
        self.loc_p3.append(dv_p3.argmax(axis=-1) + 1)
        # loc_p3 = dv_p3.argmax(axis=-1)
        # self.data_save = filtered_p3.copy()

        init_values = np.hstack((np.arange(1, 19), np.arange(109, 127)))

        self.results_predict.append(init_values[self.loc_ssvep[-1] - 1] + (self.loc_p3[-1] - 1) * 18)


if __name__ == '__main__':

    # Initialization class ReadData
    event_thread_process = threading.Event()
    read_data = ReadData(ip_address='169.254.14.131')

    n_chars = 36
    n_rounds = 5
    t_cut = 5.1 # if n_rounds == 1 else 3.7
    # [2.3, 3, 3.7, 4.4, 5.1, 5.8]
    # Initialization class ProcessingRecog
    w_pass_2d = np.array([[8, 18, 28, 38, 48, 58], [72, 72, 72, 72, 72, 72]])  # 70
    w_stop_2d = np.array([[6, 16, 26, 36, 46, 56], [74, 74, 74, 74, 74, 74]])  # 72
    w_pass_p3 = np.array([[0.5], [10]])
    w_stop_p3 = np.array([[0.1], [12]])
    process_recog = ProcessingRecog(t_begin_cls=0.14, t_end_cls=0.14+0.8, fs_down=250, CHANNELS=read_data.CHANNELS,
                                 chs_used=['cpz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'PZ', 'P1',
                                           'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3', 'PO4', 'PO5', 'PO6',
                                           'PO7', 'PO8', 'OZ', 'O1', 'O2', 'CB1', 'CB2'],
                                 w_pass_2d=w_pass_2d, w_stop_2d=w_stop_2d, event_thread_process=event_thread_process,
                                 n_points_buffer=read_data.n_points_buffer, fs_orig=read_data.fs_orig,
                                 end_flag_trial=read_data.end_flag_trial, t_begin_buffer=t_cut,
                                 chs_p3=['F3', 'Fz', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO7', 'PO8', 'OZ'],
                                 w_pass_p3=w_pass_p3, w_stop_p3=w_stop_p3, fs_down_p3=250, t_begin_p3=0.05, t_end_p3=0.8,
                                 n_rounds=n_rounds)
    process_recog.load_model()
    process_recog.Daemon = True
    process_recog.start()

    # Start
    read_data.Daemon = True

    read_data.connect_tcp()
    time.sleep(2)
    read_data.start_acq()
    time.sleep(2)
    read_data.start()

    # Udp
    send_result = SendMessageUdp(server_ip='169.254.11.113')
    send_result.start_client()

    time.sleep(2)
    stop_flag = False
    idx_iter = 0
    while not stop_flag:
        # if read_data.is_activated():
        if read_data.event_thread_read.is_set():
            read_data.event_thread_read.clear()
            # print('Current state of Reading thread：', read_data.event_thread_read.is_set())

            t_begin = time.time()
            process_recog.transmit_data(read_data._data_process, read_data._ptr_label, read_data.n_points_packet, read_data.n_points_buffer)
            process_recog.flag_process = True
            idx_iter += 1
            print('The {}-th enters the processing thread.'.format(idx_iter))

            if event_thread_process.is_set():
                event_thread_process.clear()

            event_thread_process.wait()
            event_thread_process.clear()
            read_data.event_thread_read.clear()
            # time.sleep(0.08)

            send_result.send_message(process_recog.results_predict[-1])
            print('The predicted result：{}, Consuming time: {} s.'.format(process_recog.results_predict[-1], time.time() - t_begin))
            # print('Current state of Reading thread：', read_data.event_thread_read.is_set())
            if idx_iter == n_chars:
                np.save('results_predict.npy', process_recog.results_predict)
                stop_flag = True

    time.sleep(5)
    read_data.stop_acq()

    # results_predict = np.load('results_predict.npy')
    print('breakpoint')
