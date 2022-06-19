# -*- coding:utf-8 -*-
"""
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2020-07-16
update: 2022-06
version 1.1

"""

import time
from typing import Optional, Union
from abc import ABCMeta, abstractmethod
import threading
import socket  # select
import struct
import warnings

import numpy as np
from numpy import ndarray
from scipy import signal

warnings.filterwarnings('ignore')  # or warnings.filterwarnings("default")


class BaseReadData(threading.Thread, metaclass=ABCMeta):
    """Base class: Read data from EEG device in real time.
    Actually, there is a data buffer caching data from the EEG device all the time.

    Parameters
    ----------
    fs_orig: int,
        raw sampling rate (also called as srate), which depends on device setting.

    ip_address: str,
        the IP of data acquisition computer, e.g. '192.168.36.27'. If None, automatically gets the IP address.

    dur_one_packet: float (=0.04 for NeuroScan), unit: seconds
        the time of one packet.

    time_buffer: int (default=30) | float, unit: second,
        time for the data buffer.

    end_flag_trial: the end flag of the new sample (also called as new trial)
        This end flag got by BaseReadData thread indicated that the BaseProcessingRecog thread starts.
        It is used for cutting data from data buffer.

    Variables
    ---------
    CHANNELS: list,
        the EEG channels list. The elements must be uppercase.

    Attributes
    ----------
    channels: int (default=64),
        the number of channels for data collection, which depends on device setting.

    n_points_buffer: int
        the size of data buffer.

    event_thread_read: object of threading.Event()
        mainly used to control reading thread.

    data_buffer: ndarray of shape (n_chs+1, n_pnts_buffer)
        the data buffer used for caching living data stream.

    n_points_packet: int
        the points of one packet.

    packet_data_bytes: int
        the size of one packet in unit bytes.

    current_ptr: int
        the pointer of data buffer.

    s_client: object of socket.

    flag_label: 1-D ndarray
        only for parallel port.
        flag_label[0] represents whether the tag value at last moment is low.
        flag_label[1] stores the tag value of the last moment.

    _port: int
        the TCP port, available only when the amplifier transmits data over tcpip.

    _ptr_label: int
        used for recoding location of the packet containing end_flag_trial.

    _unpack_data_fmt: str
        the format of unpacking packet.

    Notes
    -----
    The attr .packet_data_bytes and ._unpack_data_fmt may be overridden for different amplifiers.
    These values can be got from amplifier documents.
    """

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

    def __init__(self,
                 fs_orig: int = 1000,
                 ip_address: Optional[Union[None, str]] = None,
                 dur_one_packet: float = 0.04,
                 time_buffer: int = 30,
                 end_flag_trial: int = 251):

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
        self._dur_one_packet = dur_one_packet  # 0.04  # unit: second
        self.n_points_packet = int(np.round(fs_orig * self._dur_one_packet))
        self.packet_data_bytes = (self.channels + 1) * self.n_points_packet * 4
        self.current_ptr = 0
        self.s_client = None
        self.flag_label = np.array([0, 0])  # only for parallel port
        self._ptr_label = 0  # used for recoding location of the packet containing end_flag_trial.

        self._unpack_data_fmt = '<' + str((self.channels + 1) * self.n_points_packet) + 'i'  # little endian

    @abstractmethod
    def connect_tcp(self):
        """Initialize TCP and Connect with EEG device.

        Returns
        -------
        self.s_client: object of socket.

        """
        pass

    @abstractmethod
    def start_acq(self):
        """Start handshake with the amplifier to establish a reliable connection.

        """
        pass

    @abstractmethod
    def stop_acq(self):
        """Stop connection.

        """
        pass

    @abstractmethod
    def get_data(self):
        """Get a new package and Convert the format (i.e. vector) to 2-D matrix.

        Returns
        -------
        self.new_data: 2-D ndarray,
            axis 0: all EEG channels + label channel. The last row is the label channel.
            axis 1: the time points.
        """
        pass

    @abstractmethod
    def update_buffer(self):
        """Update data buffer when a new package arrived.

        """
        pass

    @abstractmethod
    def reset_buffer(self):
        """Reset data buffer.

        """
        pass

    @abstractmethod
    def is_activated(self):
        pass

    @abstractmethod
    def close_connection(self):
        pass

    def _recv_fixed_len(self, n_bytes):
        """Receive data in integer multiples of packets.

        Returns
        -------
         b_data: binary data saved in hexadecimal format.

        """
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

    def run(self):
        lock_read = threading.Lock()
        while True:
            # rs, _, _ = select.select([self.s_client], [], [], 12)  # Make sure the connection state
            # if not rs:
            #     raise ValueError('Connection Failed, the tcp/ip may be unstable.')
            if self.s_client:  # rs[0]
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


class BaseProcessingRecog(threading.Thread, metaclass=ABCMeta):
    """Base class: Processing and Recognize signal.

    Parameters
    ----------
    t_begin_cls: float, unit: second,
        the beginning time for the used time window that is relative to the stimulus onset.

    t_end_cls: float, unit: second,
        the ending time for the used time window that is relative to the stimulus onset.

    fs_down: int,
        the down-sampling rate (default: 250Hz).

    CHANNELS: list,
        channels montage, from class BaseReadData.

    chs_used: list | str,
        The channels were used for subsequent experimental analysis.
        e.g. chs_used = ['POZ','PZ','PO3','PO5','PO4','PO6','O1','OZ','O2']

    w_pass_2d: ndarray of shape 2-D,
        w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
        w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.

    w_stop_2d: ndarray of shape 2-D,
        w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
        w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])

    event_thread_process: Thread Event.

    n_points_buffer: int,
        the number of time points for data buffer, from class BaseReadData.

    fs_orig: int,
        the raw sampling rate (default 1,000 Hz for most of experiments), from class BaseReadData.

    end_flag_trial: int,
        the end flag of the new sample (aka. new trial), used for cutting data from data buffer.
        This end flag got by BaseReadData thread indicated the BaseProcessingRecog thread starts.

    t_begin_buffer: float, unit: second,
        the beginning time to cut data from buffer in relative to ending label (e.g. label value 251).

    t_end_buffer: float, unit: second,
        the ending time to cut data from buffer in relative to ending label (e.g. label value 251).
        Actually, if the amplifier is Neuroscan, this parameter can range from 0 to 0.04s (i.e. within the time of a packet).
        Experimentally, it is recommended that it sets to 0 and the reserved data length is considered by setting trigger.

    raw_data: ndarray of shape 2-D (n_chs, n_pnts),
        axis 0: all channels + label channel. The last row is the label channel.
        axis 1: the time points, which have been cut from data buffer.

    Attributes
    ----------
    n_chs: int,
        the number of channels montage.

    flag_process: bool,
        the flag of the Thread signal processing.

    results_predict: list,
        all predicted results. The last element is the latest result for the new trial.

    _ptr_label: int,
        used for recoding location of the packet containing end_flag_trial.

    raw_data: raw data, ndarray of shape (n_chs, n_pnts).
        Notably, that the data is raw, that is to say, it's not processed in any way, e.g. resample, filter, etc.
        axis 0: all EEG channels. (=len(CHANNELS))
        axis 1: the time points.

    evt: ndarray of shape (n_events, 2(i.e. value and latency))
        event corresponding to raw EEG data.
    """

    def __init__(self, t_begin_cls, t_end_cls, fs_down, CHANNELS, chs_used, w_pass_2d, w_stop_2d, event_thread_process,
                 n_points_buffer, fs_orig, end_flag_trial, t_begin_buffer=1, t_end_buffer=0, raw_data=None):

        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')

        threading.Thread.__init__(self)

        self.t_begin_cls = t_begin_cls
        self.t_end_cls = t_end_cls
        self.fs_down = fs_down
        self._CHANNELS = CHANNELS
        self.chs_used = self._select_chs(chs_used)
        self.n_chs = len(chs_used)
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

    def transmit_data(self, data_buffer, ptr_label, n_points_packet, n_points_buffer):
        """Pass and update parameters from class ReadData.

        Parameters
        ----------
        data_buffer: ndarray of shape (n_chs+1, n_pnts_buffer)
            the copy of data buffer.

        ptr_label: int,
            the pointer of data buffer.

        n_points_packet: int,
            the points of one packet.

        n_points_buffer: int,
            the size of data buffer.
        """
        if self._ptr_label == ptr_label:
            warnings.warn('\nThe result may be NOT correct. The loc of current label overlaps with previous one.')

        self.data_buffer = data_buffer.copy()
        self._ptr_label = ptr_label
        self._n_points_packet = n_points_packet
        self._n_points_buffer = n_points_buffer

    def data_from_buffer(self):
        """Cut data from buffer and Extract event (i.e. trigger).
        The flag of data ending is label value 251.

        Annotation
        ----------
        self.data_buffer: 2-D ndarray, from class BaseReadData,
            axis 0: all channels + label channel. The last row is the label channel.
            axis 1: the time points (default: 30s).

        Returns
        -------
        self.raw_data: ndarray of shape (n_chs, n_pnts).

        self.evt: event corresponding to raw EEG data.
        """
        loc_ending = np.argwhere(self.data_buffer[-1,:] == self.end_flag_trial)
        n_valid_label = loc_ending.shape[0]

        if n_valid_label != 1:
            loc_seq = []
            self._ptr_label = self._ptr_label if self._ptr_label >= self._n_points_packet else self._ptr_label+self._n_points_buffer
            for idx_loc, idx_value in enumerate(loc_ending[:, 0]):
                tmp_value = self._ptr_label - idx_value
                # the distance between the current ptr and end flag is not greater than 40.
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
        self.raw_data = self.data_buffer[:-1, np.mod(np.arange(loc_begin, loc_ending), self.n_points_buffer)]

        evt_value_buff = self.data_buffer[-1, np.mod(np.arange(loc_begin, loc_ending), self.n_points_buffer)]
        evt_latency = np.argwhere(evt_value_buff != 0)
        evt_value = evt_value_buff[evt_latency]
        self.evt = np.hstack((evt_value, evt_latency)).astype(int)  # 2-D: n_events * 2(i.e. value and latency)

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

        Returns
        -------
        raw_data_resample: 2-D ndarray, the resampled data.
            axis 0: all EEG channels.
            axis 1: the time points.
        events: 2-D ndarray, all event values and latencies.
            n_events * 2(i.e. value and latency).
        """
        n_points = raw_data.shape[1]
        fs_down = self.fs_down if len(args) == 0 else args[0]
        # fs_down, *_= args

        if self.fs_orig > fs_down:
            events = np.zeros_like(self.evt)
            raw_data_resample = signal.resample(raw_data, int(np.ceil(fs_down * n_points / self.fs_orig)), axis=1)
            # an inevitable loss of precision, but the impact is negligible.
            events[:, 0], events[:, -1] = self.evt[:, 0], (self.evt[:, -1]/(self.fs_orig/fs_down)).astype(int)
            return raw_data_resample, events

        elif self.fs_orig == fs_down:
            # self.raw_data is raw_data_resample.
            return raw_data, self.evt

        else:
            raise ValueError('Oversampling is NOT recommended. The reason is self.fs < self.fs_down.')

    def filtered_data_iir(self, raw_data, *args):
        """Demo returned filtered_data is dict.

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
        filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chs * n_samples * n_classes * n_trials.
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
        filtered_data = dict()
        n_filters = w_pass.shape[1]
        for idx_filter in range(n_filters):
            sos_system['filter'+str(idx_filter+1)] = \
                self._get_iir_sos_band([w_pass[0, idx_filter], w_pass[1, idx_filter]],
                                       [w_stop[0, idx_filter], w_stop[1, idx_filter]], fs_down)
            filtered_data['bank'+str(idx_filter+1)] = \
                signal.sosfiltfilt(sos_system['filter'+str(idx_filter+1)], raw_data, axis=-1)

        return filtered_data

    def filtered_data_iir_2(self, raw_data, *args):
        """Demo returned filtered_data is ndarray, which is convenient for the following matrix manipulation.
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

    @abstractmethod
    def recognition(self):
        '''Recognize the class label for the new sample.

        Returns
        -------


        Notes
        -----
        Loading model established in the offline experiments is needed if the task is a training problem.

        The following key parameters may be used.
        ------
        self.raw_data: EEG data, 2-D ndarray.
            Notely, that the data is raw, that is to say, it's not processed in any way, e.g. resample, filter, etc.
            axis 0: all EEG channels. (=len(CHANNELS))
            axis 1: the time points.

        self.evt: event corresponding to raw EEG data.
        '''
        pass

    def run(self):
        while True:
            if not self.flag_process:
                continue
            else:
                t1 = time.time()
                self.data_from_buffer()
                self.recognition()  # self.results_predict[-1]
                self.flag_process = False
                # time.sleep(2)
                self.event_thread_process.set()
                print('Processing Thread: Consuming time: %.6f s' % (time.time() - t1))


class SendMessageUdp():

    def __init__(self, server_ip, server_port=9094, client_ip=None, client_port=9095):

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
