# -*- coding:utf-8 -*-
'''
@ author: Orion Han
@ email: jinhan@tju.edu.cn
@ Created on: 2020-07-18
version 1.0

Application: xxxxxx

'''
import time
import threading
import warnings

import numpy as np
from scipy import signal
import scipy.io as sio

from BaseFrame_OnlineBCI import BaseReadData
from BaseFrame_OnlineBCI import BaseProcessRecog
from BaseFrame_OnlineBCI import SendMessageUdp
from algorithms import corr_coeff_manu

warnings.filterwarnings('ignore')  # or warnings.filterwarnings("default")

# Custom Parameters
end_flag_trial = 222  # 1-255
ip_address = None  # If None, automatically get ip address. or '127.0.0.1'
channels = 68  # MUST be same with len(ReadData.CHANNELS)


class ReadData(BaseReadData):

    # also can override CHANNELS

    def __init__(self):
        super().__init__(fs_orig=1000, ip_address=ip_address, time_buffer=30, end_flag_trial=end_flag_trial)


class ProcessRecog(BaseProcessRecog):

    def __init__(self, t_begin_cls, t_end_cls, fs_down, CHANNELS, chan_used, w_pass_2d, w_stop_2d, event_thread_process,
                 n_points_buffer, fs_orig, end_flag_trial, t_begin_buffer, flag_process=False):
        super().__init__(t_begin_cls, t_end_cls, fs_down, CHANNELS, chan_used, w_pass_2d, w_stop_2d, event_thread_process,
                         n_points_buffer, fs_orig, end_flag_trial, t_begin_buffer, t_end_buffer=0, raw_data=None)
        self.flag_process = flag_process

    def filtered_data_iir(self, raw_data_resample):
        '''
        :param raw_data_resample: 2-D ndarray, the resampled data from method resample_data.
            axis 0: all EEG channels.
            axis 1: the time points.
        :return:
            self.num_filter: num of filter (for filter bank)
            filtered_data: 3-D, numpy, n_chans * n_points * n_filters
        '''

        sos_system = dict()
        self.num_filter = self.w_pass_2d.shape[1]
        filtered_data = np.empty((self.n_chans, raw_data_resample.shape[1], self.num_filter))
        for idx_filter in range(self.num_filter):
            sos_system['filter'+str(idx_filter+1)] = self._get_iir_sos_band(w_pass=[self.w_pass_2d[0, idx_filter], self.w_pass_2d[1, idx_filter]],
                                                                            w_stop=[self.w_stop_2d[0, idx_filter],
                                                                                    self.w_stop_2d[1, idx_filter]])
            filtered_data[:, :, idx_filter] = \
                signal.sosfiltfilt(sos_system['filter'+str(idx_filter+1)], raw_data_resample, axis=1)

        return filtered_data

    def filtered_data_iir_2(self, raw_data_resample):
        '''
        demo when filtered_data is dict.
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        Generate self.num_filter
        '''

        sos_system = dict()
        filtered_data = dict()
        self.num_filter = self.w_pass_2d.shape[1]
        for idx_filter in range(self.num_filter):
            sos_system['filter'+str(idx_filter+1)] = self._get_iir_sos_band(w_pass=[self.w_pass_2d[0, idx_filter], self.w_pass_2d[1, idx_filter]],
                                                                            w_stop=[self.w_stop_2d[0, idx_filter],
                                                                                    self.w_stop_2d[1, idx_filter]])
            filtered_data['bank'+str(idx_filter+1)] = \
                                                signal.sosfiltfilt(sos_system['filter'+str(idx_filter+1)], raw_data_resample, axis=1)

        return filtered_data

    def ext_epochs(self, filtered_data):
        '''

        :param filtered_data: 3-D, numpy, n_chans * n_points * n_filters.
        :param self.evt: 2-D ndarray, from method channel_selected of base class BaseProcessRecog.
                format - n_events * 2(i.e. value and latency)
        :return: new_epoch: 3-D,  n_chans * n_points * n_filters, e.g. (9, 126, 7)
        '''
        # TODO：the condition is waited to update.
        if self.evt.shape[0] != 2:
            raise ValueError('The row of self.evt should be equal to 1.')

        t_trigger = self.evt[0, 1]
        latency_begin = int(round(self.fs_down * self.t_begin_cls) + t_trigger)
        latency_end = int(round(self.fs_down * self.t_end_cls) + t_trigger)

        new_epoch = filtered_data[:, latency_begin:latency_end + 1, :]

        if new_epoch.shape[1] == 125:
            print('problems')

        return new_epoch

    def fusion_coef(self):
        weight_a = np.zeros(self.num_filter)
        for idx_filter in range(self.num_filter):
            idx_filter += 1
            weight_a[idx_filter-1] = idx_filter**(-1.25)+0.25

        return weight_a

    def load_model(self):
        # load training model
        training_model = sio.loadmat(r'model.mat')
        # w_pro.shape: (9, 40, 7)
        # x_template.shape: (9, 126, 40, 7)
        self.w_pro, self.x_template = training_model['w_pro'], training_model['x_template']
        self.n_class = self.w_pro.shape[1]

    def recognition(self, new_epoch):

        rr_coeff = np.zeros((self.num_filter, self.n_class))
        for idx_filter in range(self.num_filter):
            test_temp = self.w_pro[:, :, idx_filter].T.dot(new_epoch[:, :, idx_filter]).reshape((1, -1), order='C')
            for idx_class in range(self.n_class):
                temp2 = self.w_pro[:, :, idx_filter].T.dot(self.x_template[:, :, idx_class, idx_filter]).reshape((1, -1), order='C')
                rr_coeff[idx_filter, idx_class] = corr_coeff_manu(test_temp, temp2)

        if self.num_filter > 1:
            rr_coeff = rr_coeff ** 2
            weight_a = self.fusion_coef()
            for idx_filter in range(self.num_filter):
                rr_coeff[idx_filter, :] *= weight_a[idx_filter]

        rr_coeff = rr_coeff.sum(axis=0, keepdims=False)

        self.results_predict.append(rr_coeff.argmax(axis=0) + 1)


    def run(self):
        while True:
            if not self.flag_process:
                continue
            else:
                self.data_from_buffer()
                self.channel_selected()
                filtered_data = self.filtered_data_iir(self.resample_data())
                new_epoch = self.ext_epochs(filtered_data)
                self.recognition(new_epoch)  # self.results_predict[-1]
                self.flag_process = False
                # time.sleep(2)
                self.event_thread_process.set()


if __name__ == '__main__':

    # Initialization class ReadData
    event_thread_process = threading.Event()
    read_data = ReadData()


    # Initialization class ProcessRecog
    w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54], [70, 70, 70, 70, 70, 70, 70]])
    w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52], [72, 72, 72, 72, 72, 72, 72]])
    # w_pass_2d = np.array([[5], [70]])
    # w_stop_2d = np.array([[3], [72]])
    process_recog = ProcessRecog(t_begin_cls=0.14, t_end_cls=0.64, fs_down=250, CHANNELS = read_data.CHANNELS,
                                 chan_used=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
                                 w_pass_2d=w_pass_2d, w_stop_2d=w_stop_2d, event_thread_process=event_thread_process,
                                 n_points_buffer=read_data.n_points_buffer, fs_orig=read_data.fs_orig,
                                 end_flag_trial=read_data.end_flag_trial, t_begin_buffer=0.8)
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
    send_result = SendMessageUdp(server_ip='192.168.223.136')
    send_result.start_client()

    time.sleep(2)
    stop_flag = False
    idx_iter = 0
    while not stop_flag:
        # if read_data.is_activated():
        if read_data.event_thread_read.is_set():
            read_data.event_thread_read.clear()
            # print('当前读取线程状态1：', read_data.event_thread_read.is_set())

            t_begin = time.time()
            process_recog.transmit_data(read_data._data_process, read_data._ptr_label, read_data.n_points_packet, read_data.n_points_buffer)
            process_recog.flag_process = True
            idx_iter += 1
            print('第{}次进入处理程序.'.format(idx_iter))

            if event_thread_process.is_set():
                event_thread_process.clear()

            event_thread_process.wait()
            event_thread_process.clear()
            read_data.event_thread_read.clear()
            # time.sleep(0.08)

            send_result.send_message(process_recog.results_predict[-1])
            print('分类结果为：{}, 耗时{} seconds.'.format(process_recog.results_predict[-1], time.time() - t_begin))
            # print('当前读取线程状态2：', read_data.event_thread_read.is_set())
            if idx_iter == 200:
                stop_flag = True

    read_data.stop_acq()
    print('breakpoint')





