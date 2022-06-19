# -*- coding:utf-8 -*-
"""
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2020-07-16
update: 2022-06
version 1.1

Application: Read live data streaming from Neuroscan amplifier.

"""

import time
from abc import ABCMeta, abstractmethod
import threading
import socket  # select
import struct
import warnings

import numpy as np
from scipy import signal

from BaseFramework import BaseReadData

warnings.filterwarnings('ignore')  # or warnings.filterwarnings("default")


class ReadNeuroscan(BaseReadData):
    """Read data stream for nueuroscan amplifier.
    The more parameter annotation can see baseclass BaseReadData.
    """

    def __init__(self, fs_orig=1000, ip_address=None, dur_one_packet=0.04, time_buffer=30, end_flag_trial=251):
        super().__init__(fs_orig, ip_address, dur_one_packet, time_buffer, end_flag_trial)


    def connect_tcp(self):

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
                print('Current recv buffer size is {} bytes, send buff size is {} bytes.'.format(buff_size_recv,
                                                                                                 buff_size_send))
                break
            except:
                print('The {}-th Connection fails, Please check params (e.g. IP address).'.format(i + 1))
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
        time.sleep(0.01)
        self.s_client.send(struct.pack('12B', 67, 84, 82, 76, 0, 2, 0, 2, 0, 0, 0, 0))  # stop acq
        self.s_client.send(struct.pack('12B', 67, 84, 82, 76, 0, 1, 0, 2, 0, 0, 0, 0))  # close connection
        self.s_client.close()

    def get_data(self):
        """Get a new package and Convert the format (i.e. vector) to 2-D matrix.

        Returns
        -------
        self.new_data: 2-D ndarray,
            axis 0: all EEG channels + label channel. The last row is the label channel.
            axis 1: the time points.
        """

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
        """Update data buffer when a new package arrived.
        """

        self.data_buffer[:, np.mod(np.arange(self.current_ptr,
                                             self.current_ptr + self.n_points_packet),
                                   self.n_points_buffer)] = self.new_data
        self.current_ptr = np.mod(self.current_ptr + self.n_points_packet, self.n_points_buffer)

        if np.any(self.new_data[-1, :] == self.end_flag_trial):  # check whether the new packet has end_flag_trial.
            self._ptr_label = self.current_ptr.copy()
            self._data_process = self.data_buffer.copy()
            self.event_thread_read.set()

    def reset_buffer(self):

        self.data_buffer = np.zeros((self.channels + 1, self.n_points_buffer))  # data buffer
        self.current_ptr = 0

    def is_activated(self):
        # return np.any(self.new_data[-1, :] == self.end_flag_trial)
        pass

    def close_connection(self):
        self.s_client.close()

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
