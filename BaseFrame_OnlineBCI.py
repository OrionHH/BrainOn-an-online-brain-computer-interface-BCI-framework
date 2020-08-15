# -*- coding:utf-8 -*-
'''
@ author: Orion Han
@ email: jinhan@tju.edu.cn
@ Created on: 2020-07-16
version 1.0

Application: Online BCI framework with multi-thread.

This online framework was developed by Jin Han, who is a student from the Lab of Neural Engineering & Rehabilitation,
Tianjin University, China.
'''

import os, time
from abc import ABCMeta, abstractmethod
import threading
import socket, select
import struct
import warnings

import numpy as np
from scipy import signal

warnings.filterwarnings('ignore')  # or warnings.filterwarnings("default")

class BaseReadData(threading.Thread, metaclass=ABCMeta):
    '''
    Read data from EEG device in real time.
    Actually, there is a data buffer that caches data from the EEG device all the time.
    '''
    CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
        'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
        'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2',
        'HEO', 'VEO', 'EKG', 'EMG'
    ]  # M1: 33. M2: 43.

    def __init__(self, fs_orig=1000, ip_address=None, time_buffer=30, end_flag_trial=251):
        '''
        :param fs_orig: int,
            original sampling rate (also called as srate), which depends on device setting.
        :param time_buffer: int (default=30) | float, unit: second,
            time for the data buffer.
        :param channels: int (default=64),
            the number of channels for data collection, which depends on device setting.
        :param ip_address: str,
            the IP of data acquisition computer, e.g. '192.168.36.27'. If None, automatically gets the IP address.
        :param dur_one_packet: float (=0.04 for NeuroScan),
            the time of one packet.
        :param current_ptr: int,
            the pointer of data buffer.
        :param end_flag_trial: the ending flag of the new sample (also called as new trial)
            This end flag got by BaseReadData thread indicated that the BaseProcessRecog thread starts.
            It is used for cutting data from data buffer.
        '''
        threading.Thread.__init__(self)

        self.fs_orig = fs_orig
        self.channels = len(self.CHANNELS)
        self.time_buffer = time_buffer
        self.n_points_buffer = int(np.round(fs_orig * time_buffer))
        self.ip_address = socket.gethostbyname(socket.gethostname()) if ip_address is None else ip_address
        self.end_flag_trial = end_flag_trial
        self.event_thread_read = threading.Event()
        self.event_thread_read.clear()
        self.data_buffer = np.zeros((self.channels + 1, self.n_points_buffer))  # data buffer

        self._port = 4000
        self._dur_one_packet = 0.04  # unit: second
        self.n_points_packet = int(np.round(fs_orig * self._dur_one_packet))
        self.packet_data_bytes = (self.channels+1) * self.n_points_packet * 4
        self.current_ptr = 0
        self.s_client = None
        # flag_label[0] represents whether the tag value at last moment is low. flag_label[1] stores the tag value of the last moment.
        self.flag_label = np.array([0, 0])
        self._ptr_label = 0  # used for recoding location of the packet containing end_flag_trial.

        self._unpack_data_fmt = '>' + str((self.channels + 1) * self.n_points_packet) + 'i'  # big endian

    def connect_tcp(self):
        '''
        Initialize TCP and Connect with EEG device.
        :return:
            self.s_client: object of socket.
        '''
        self.s_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        SEND_BUF_SIZE = self.packet_data_bytes  # unit: bytes
        RECV_BUF_SIZE = self.packet_data_bytes * 9  # unit: bytes
        time_connect = time.time()
        for i in range(5):
            try:
                time.sleep(1.5)
                self.s_client.connect((self.ip_address, self._port))
                print('Connect Successfully.')
                # Get current size of the socket's send buffer
                # buff_size = self.s_client.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)  # 8192
                self.s_client.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
                self.s_client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
                self.s_client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)

                buff_size_send = self.s_client.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                buff_size_recv = self.s_client.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                print('Current recv buffer size is {} bytes, send buff size is {} bytes.'.format(buff_size_recv, buff_size_send))
                break
            except:
                print('The {}-th Connection fails, Please check params (e.g. IP address).'.format(i+1))
                if i == 4:
                    print('The %s-th Connection fails, program exits.')
                    time_connect = time.time() - time_connect
                    print('Consuming time of Connection is {:.4f} seconds.'.format(time_connect))
                    self.s_client.close()

    def start_acq(self):
        # start_acq_command: 67, 84, 82, 76, 0, 2, 0, 1, 0, 0, 0, 0
        # start_get_command: 67, 84, 82, 76, 0, 3, 0, 3, 0, 0, 0, 0
        # start collecting data
        self.s_client.send(struct.pack('12B', 67, 84, 82, 76, 0, 2, 0, 1, 0, 0, 0, 0))  # start acq
        header_packet = self._recv_fixed_len(24)
        # start getting data
        print('Start getting data from buffer by TCP/IP.')
        self.s_client.send(struct.pack('12B', 67, 84, 82, 76, 0, 3, 0, 3, 0, 0, 0, 0))  # start get data

    def stop_acq(self):
        self.s_client.send(struct.pack('12B', 67, 84, 82, 76, 0, 3, 0, 4, 0, 0, 0, 0))  # stop getting data
        time.sleep(0.001)
        self.s_client.send(struct.pack('12B', 67, 84, 82, 76, 0, 2, 0, 2, 0, 0, 0, 0))  # stop acq
        self.s_client.send(struct.pack('12B', 67, 84, 82, 76, 0, 1, 0, 2, 0, 0, 0, 0))  # close connection
        self.s_client.close()

    def get_data(self):
        '''
        Get a new package and Convert the format (i.e. vector) to 2-D matrix.
        :return: self.new_data: 2-D ndarray,
            axis 0: all EEG channels + label channel. The last row is the label channel.
            axis 1: the time points.
        '''
        tmp_header = self._recv_fixed_len(12)
        details_header = self._unpack_header(tmp_header)
        
        if details_header[-1] != self.packet_data_bytes:
            raise ValueError('The .ast template is not matched with class Variable CHANNELS. Please RESET CHANNELS.')

        # 2-D: (EEG channels + label channel) * time points (i.e. =40 for 1000Hz sampling rate)
        bytes_data = self._recv_fixed_len(self.packet_data_bytes)
        new_data_trans = self._unpack_data(bytes_data)
        new_data_temp = np.empty(new_data_trans.shape, dtype=np.float)
        new_data_temp[:-1, :] = new_data_trans[:-1, :] * 0.0298  # unit: μV

        # verify valid label
        new_data_temp[-1, :] = np.zeros(new_data_trans.shape[1])
        loc_label = np.arange(self.channels * 4, self.packet_data_bytes, (self.channels + 1) * 4)
        if len(loc_label) != new_data_trans.shape[1]:
            raise ValueError('An Error occurred, generally because the .ast template is not matched with CHANNELS.')

        for idx_time, loc_bytes in enumerate(loc_label):
            label_value = bytes_data[loc_bytes]
            if label_value != 0 and self.flag_label[0] == 0:  # rising edge of TTL voltage
                self.flag_label[0] = 1
                self.flag_label[1] = label_value
                new_data_temp[-1, idx_time] = label_value
            elif label_value != 0 and self.flag_label[0] == 1 and self.flag_label[1] == label_value:
                new_data_temp[-1, idx_time] = 0
            elif label_value == 0 and self.flag_label[0] == 1:
                self.flag_label[0] = 0

        self.new_data = new_data_temp

    def update_buffer(self):
        '''
        Update data buffer when a new package arrived.
        '''
        self.data_buffer[:,np.mod(np.arange(self.current_ptr,
                                            self.current_ptr+self.n_points_packet), self.n_points_buffer)] = self.new_data
        self.current_ptr = np.mod(self.current_ptr + self.n_points_packet, self.n_points_buffer)

        if np.any(self.new_data[-1, :] == self.end_flag_trial):  # check whether the new packet has end_flag_trial.
            self._ptr_label = self.current_ptr.copy()
            self._data_process = self.data_buffer.copy()
            self.event_thread_read.set()

    def reset_buffer(self):
        '''
        Reset data buffer.
        '''
        self.data_buffer = np.zeros((self.channels + 1, self.n_points_buffer))  # data buffer
        self.current_ptr = 0

    def is_activated(self):
        # return np.any(self.new_data[-1, :] == self.end_flag_trial)
        pass

    def close_connection(self):
        self.s_client.close()

    def _recv_fixed_len(self, n_bytes):
        b_data = b''
        flag_stop_recv = False
        b_count = 0
        while not flag_stop_recv:
            try:
                tmp_bytes = self.s_client.recv(n_bytes - b_count)
            except socket.timeout:
                raise ValueError('No data is Getted.')

            if b_count == n_bytes or not tmp_bytes:
                flag_stop_recv = True

            b_count += len(tmp_bytes)
            b_data += tmp_bytes

        return b_data

    def _unpack_header(self, header_packet):
        # header for a packet
        chan_name = struct.unpack('>4s', header_packet[:4])
        w_code = struct.unpack('>H', header_packet[4:6])
        w_request = struct.unpack('>H', header_packet[6:8])
        packet_size = struct.unpack('>I', header_packet[8:])

        return (chan_name[0].decode('utf-8'), w_code[0], w_request[0], packet_size[0])

    def _unpack_data(self, data_packet):
        # data for a packet, bytes stream
        data_trans = np.asarray(struct.unpack(self._unpack_data_fmt, data_packet)).reshape((-1, self.channels + 1)).T

        return data_trans

    def run(self):
        lock_read = threading.Lock()
        while True:
            # rs, _, _ = select.select([self.s_client], [], [], 12)  # Make sure the connection state
            # if not rs:
            #     raise ValueError('Connection Failed, the tcp/ip may be unstable.')
            if self.s_client:  # rs[0] ==
                lock_read.acquire()
                # t1 = time.time()
                try:
                    self.get_data()
                except:
                    print('Some problems have arisen, can not receive data from socket.')
                    lock_read.release()
                    self.s_client.close()
                else:
                    self.update_buffer()
                    # print('Consuming time to get a packet is {:.4f} ms.'.format((time.time() - t1) * 1000))
                    lock_read.release()


class BaseProcessRecog(threading.Thread, metaclass=ABCMeta):

    def __init__(self, t_begin_cls, t_end_cls, fs_down, CHANNELS, chan_used, w_pass_2d, w_stop_2d, event_thread_process,
                 n_points_buffer, fs_orig, end_flag_trial, t_begin_buffer=1, t_end_buffer=0, raw_data=None):
        '''
        Initialization parameters.
        :param t_begin_cls: float, unit: second,
            It is the starting time for time window used, and is relative to the stimulus onset.
        :param t_end_cls: float, unit: second,
            It is the ending time for time window used, and is relative to the stimulus onset.
        :param fs_down: int,
            the down-sampling rate (default: 250Hz).
        :param CHANNELS: list, from class BaseReadData
        :param chan_used: list | str,
            The channels were used for subsequent experimental analysis.
            e.g. chan_used = ['POZ','PZ','PO3','PO5','PO4','PO6','O1','OZ','O2']
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
            e.g.
            w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
            w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        :param event_thread_process: Thread Event.
        :param n_points_buffer: int, from class BaseReadData,
            the number of time points of data buffer.
        :param fs_orig: int, from class BaseReadData,
            the original sampling rate (default 1,000 Hz for most of experiments)
        :param end_flag_trial: the ending flag of the new sample (also called as new trial)
            This end flag got by BaseReadData thread indicated that the BaseProcessRecog thread starts.
            It is used for cutting data from data buffer.
        :param t_begin_buffer: float, unit: second,
            It is the starting time to cut data from buffer in relative to ending label (e.g. label value 251).
        :param t_end_buffer: float, unit: second,
            It is the ending time to cut data from buffer in relative to ending label (e.g. label value 251).
            Actually, this parameter ranges from 0 to 0.04s (i.e. within the time of a packet).
        :param raw_data: 2-D nadrray,
            axis 0: all channels + label channel. The last row is the label channel.
            axis 1: the time points, which have been cut from data buffer.
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')

        threading.Thread.__init__(self)

        self.t_begin_cls = t_begin_cls
        self.t_end_cls = t_end_cls
        self.fs_down = fs_down
        self._CHANNELS = CHANNELS
        self.chan_used = chan_used
        self.n_chans = len(chan_used)
        self.w_pass_2d = w_pass_2d
        self.w_stop_2d = w_stop_2d
        self.event_thread_process = event_thread_process

        self.n_points_buffer = n_points_buffer
        self.fs_orig = fs_orig
        self.t_begin_buffer = t_begin_buffer
        self.t_end_buffer = t_end_buffer
        self.raw_data = raw_data
        self.end_flag_trial = end_flag_trial
        self.flag_process = False
        self.results_predict = []
        self._ptr_label = 0

    def transmit_data(self, data_buffer, ptr_label, n_points_packet, n_points_buffer):
        '''
        Pass and update parameters from class ReadData.
        :param data_buffer:
        :param ptr_label:
        :return:
        '''
        if self._ptr_label == ptr_label:
            warnings.warn('\nThe result may be NOT correct. The loc of current label overlaps with previous one.')

        self.data_buffer = data_buffer.copy()
        self._ptr_label = ptr_label
        self._n_points_packet = n_points_packet
        self._n_points_buffer = n_points_buffer

    def data_from_buffer(self):
        '''
        Cut data from buffer.
        The flag of data ending is label value 251.
        :param self.data_buffer: 2-D ndarray, from class BaseReadData,
            axis 0: all channels + label channel. The last row is the label channel.
            axis 1: the time points (default: 30s).
        :return: self.raw_data
            the format is same with method __init__ of class BaseProcessRecog.
        '''
        loc_ending = np.argwhere(self.data_buffer[-1,:] == self.end_flag_trial)
        n_valid_label = loc_ending.shape[0]

        if n_valid_label != 1:
            loc_seq = []
            self._ptr_label = self._ptr_label if self._ptr_label >= self._n_points_packet else self._ptr_label + self._n_points_buffer
            for idx_loc, idx_value in enumerate(loc_ending[:, 0]):
                tmp_value = self._ptr_label - idx_value
                if tmp_value > 0 and tmp_value <= 40:
                    loc_seq.append(loc_ending[idx_loc, 0])

            if len(loc_seq) != 1:
                if len(loc_seq) == 0:
                    raise ValueError('No end flag is detected, which is impossible.')
                else:
                    raise ValueError('Multi end flags are detected, which is illogical.')
            else:
                loc_ending = loc_seq[0]

        loc_begin = np.mod(loc_ending - int(np.round(self.fs_orig * self.t_begin_buffer)), self.n_points_buffer) + 1
        loc_ending = loc_begin + int(np.round(self.fs_orig * (self.t_end_buffer + self.t_begin_buffer)))

        self.raw_data = self.data_buffer[:, np.mod(np.arange(loc_begin, loc_ending + 1), self.n_points_buffer)]

    def channel_selected(self):
        '''
        Select channel to use for the next processing step.
        :return:
            self.raw_data:
                the format is same with method __init__ of class BaseProcessRecog.
            self.evt: 2-D ndarray,
                format - n_events * 2(i.e. value and latency)
        '''
        idx_loc = list()
        if isinstance(self.chan_used, list):
            for _, char_value in enumerate(self.chan_used):
                idx_loc.append(self._CHANNELS.index(char_value.upper()))

        evt_value_buff = self.raw_data[-1,:]
        evt_latency = np.argwhere(evt_value_buff != 0)
        evt_value = evt_value_buff[evt_latency]

        self.evt = np.hstack((evt_value, evt_latency))  # 2-D: n_events * 2(i.e. value and latency)
        self.raw_data = self.raw_data[idx_loc, :]

    def resample_data(self):
        '''
        Down-sampling data from self.fs_orig to self.fs_down Hz.
        :return:
            raw_data_resample: 2-D ndarray, the resampled data.
                axis 0: all EEG channels.
                axis 1: the time points.
            self.evt: 2-D ndarray, all event values and latencies.
                n_events * 2(i.e. value and latency).
        '''
        n_points = self.raw_data.shape[1]
        if self.fs_orig > self.fs_down:
            # TODO 原因在这
            raw_data_resample = signal.resample(self.raw_data, int(np.ceil(self.fs_down * n_points / self.fs_orig)), axis=1)
            # TODO: latency UNTEST
            self.evt[:, -1] = np.round(self.evt[:, -1]/(self.fs_orig/self.fs_down))
            return raw_data_resample
        elif self.fs_orig == self.fs_down:
            # self.raw_data is raw_data_resample.
            return self.raw_data
        else:
            raise ValueError('Oversampling is NOT recommended. The reason is self.fs < self.fs_down.')

    def _get_iir_sos_band(self, w_pass, w_stop):
        '''
        Get second-order sections (like 'ba') of Chebyshev type I filter for band-pass.
        :param w_pass: list, 2 elements, e.g. [5, 70]
        :param w_stop: list, 2 elements, e.g. [3, 72]
        :return: sos_system
            i.e the filter coefficients.
        '''
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        wp = [2 * w_pass[0] / self.fs_down, 2 * w_pass[1] / self.fs_down]
        ws = [2 * w_stop[0] / self.fs_down, 2 * w_stop[1] / self.fs_down]
        gpass = 4  # it's -3dB when setting as 3.
        gstop = 30  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

        return sos_system

    @abstractmethod
    def filtered_data_iir(self, raw_data_resample):
        '''
        Filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param raw_data_resample: 2-D ndarray, the resampled data from method resample_data.
                axis 0: all channels + label channel. The last row is the label channel.
                axis 1: the time points.
        :return: filtered_data: dict | 3-D ndarray (recommend)
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 2-D, numpy, n_chans * time points(n_points).
            3-D, numpy, n_chans * time points(n_points) * n_filters (if filter bank uses.)
        '''
        pass

    @abstractmethod
    def ext_epochs(self, filtered_data):
        '''
        Extract epochs according to self.evt, self.t_begin_cls, and self.t_end_cls.
        :param filtered_data: from method filtered_data_iir
        :param self.evt: 2-D ndarray, all event values and latencies.
                n_events * 2(i.e. value and latency).
        :return: new_epoch: the format is same as filtered_data.
        '''
        pass

    @abstractmethod
    def recognition(self, data_epoch):
        '''
        Recognize the class label of the new sample.
        load model, which is established in the offline experiments, is needed if the task is a training problem.
        :return:
        '''
        pass

    @abstractmethod
    def run(self):
        pass


class SendMessageUdp():

    def __init__(self, server_ip, server_port=23333, client_ip=None, client_port=23332):

        self.dest_ip = server_ip
        self.dest_port = server_port
        self.source_ip = socket.gethostbyname(socket.gethostname()) if client_ip is None else client_ip
        self.source_port = client_port

    def start_client(self):
        self.sock_client = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.sock_client.bind((self.source_ip, self.source_port))

    def send_message(self, message):
        if isinstance(message, bytes):
            self.sock_client.sendto(message, (self.dest_ip, self.dest_port))
        else:
            try:
                message = struct.pack('B', message)
            except TypeError as err:
                raise TypeError(err.args)
            else:
                self.sock_client.sendto(message, (self.dest_ip, self.dest_port))

    def close_connect(self):
        self.sock_client.close()


if __name__ == '__main__':
    # 1. Test class BaseReadData
    # 1.1 Simulate and Generate a new packet of EEG device
    '''
    Firstly, execute simulate_serve_new_packet.py
    '''
    # 1.2 read data in real time.
    read_data = BaseReadData(fs_orig = 1000, ip_address = '127.0.0.1', time_buffer = 30, end_flag_trial=251)
    read_data.connect_tcp()
    iter_loop = 0
    while True:
        read_data.get_data()
        read_data.update_buffer()
        iter_loop += 1
        if iter_loop == (read_data.time_buffer // read_data.dur_one_packet) + 6:
            print('Data Buffer is full and then updates by a few more packets.')
            read_data.close_connection()
            break

    # Test class BaseProcessRecog
    chan_used = ['POZ','PZ','PO3','PO5','PO4','PO6','O1','OZ','O2']
    pr_ml = BaseProcessRecog(t_begin_cls=0.14, t_end_cls=0.64, fs_down=250, chan_used=chan_used, n_points_buffer=30000, fs_orig=1000,
                         t_begin_buffer=1, t_end_buffer=0, raw_data=None, end_flag_trial=251)  # create process and recognition thread
    print('Current thread is %s' % threading.current_thread().getName())

    pr_ml.start()




