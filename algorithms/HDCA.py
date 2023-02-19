# -*- coding:utf-8 -*-
"""
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2022.07
Refer:
    [1] Sajda, Paul, et al. "In a blink of an eye and a switch of a transistor: cortically coupled computer vision."
        Proceedings of the IEEE 98.3 (2010): 462-478.

Application: Binary problems, e.g., single trial recognition for P300-based BCI.

"""
import warnings

import numpy as np
from numpy import ndarray
from scipy import linalg as LA
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from .STDA import LDA_kernel


class HDCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Hierarchical Discriminant Component Analysis (HDCA)

    Parameters
    ----------
    fs: int, unit: Hz.
        sampling rate for the input data.

    t_dur: float, unit: second.
         assume the stationary time T of approximately 100 ms in the paper.

    Attributes
    ----------
    win_len: int,
        the number of data points for the time window (i.e., stationary time in the paper).

    win_num: int,
        the number of time windows that can be divided for the features of time dimension.

    win_start_pnt: ndarray of shape (win_num, )
        the start locations for each the divided time window.

    spatial_w: list, length = win_num.
        spatial filters for all time windows. Each element for list is a ndarray of shape (1, n_chs).
        It is the first step for HDCA.

    time_w: ndarray of shape (1, win_num)
        time filter that is the second step for HDCA.

    n_chs: int,
        the number of channels for the training set.

    n_features: int,
        the number of time points for the training set.

    """

    def __init__(self, fs, t_dur=0.1):

        self.fs = int(fs)
        self.t_dur = t_dur

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: array-like of shape (n_samples, n_chs, n_features)
           Training data.

        y : array-like of shape (n_samples,)
            Target values, {-1, 1} or {0, 1}.
        """
        self.classes_ = np.unique(y)
        _, self.n_chs, self.n_features = X.shape
        n_classes = len(self.classes_)

        # construct time window sequences
        self.win_len = int(self.fs * self.t_dur)
        if (self.n_features % self.win_len) != 0:
            warnings.warn('The stationary time (i.e., t_dur) is not recommended, '
                          'due to discarding some features of the tail of feature vector if in this way.')

            warnings.warn('The reason for above problem is that n_features % int(self.fs * self.t_dur) is NOT an Integer.')
        self.win_start_pnt = np.arange(0, self.n_features-1, self.win_len)
        self.win_num = len(self.win_start_pnt)

        # Extract samples of two classes
        loc = [np.argwhere(y == self.classes_[idx_class]).squeeze() for idx_class in range(n_classes)]
        X1, X2 = X[loc[0]].transpose([1, 2, 0]), X[loc[1]].transpose([1, 2, 0])  # X1: negative samples. X2: positive samples.

        n_samples_c1, n_samples_c2 = X1.shape[-1], X2.shape[-1]

        self.spatial_w = []
        c1_spatial_proj, c2_spatial_proj = np.zeros((n_samples_c1, self.win_num)), np.zeros((n_samples_c2, self.win_num))
        for idx_win in range(self.win_num):
            win_seq = np.arange(self.win_start_pnt[idx_win], self.win_start_pnt[idx_win]+self.win_len)
            c1_win_data, c2_win_data = X1[:, win_seq, :], X2[:, win_seq, :]

            # Notably, the output spatial_w is corresponding to positive samples.
            spatial_filter, _ = LDA_kernel(c2_win_data.reshape((self.n_chs, -1)).T, c1_win_data.reshape((self.n_chs, -1)).T)

            c1_spatial_proj[:, idx_win] = np.matmul(spatial_filter[np.newaxis,...], c1_win_data.transpose([2, 0, 1])).sum(axis=-1).squeeze()
            c2_spatial_proj[:, idx_win] = np.matmul(spatial_filter[np.newaxis,...], c2_win_data.transpose([2, 0, 1])).sum(axis=-1).squeeze()

            self.spatial_w.append(spatial_filter)

        # time filter
        self.time_w, self.threshold = LDA_kernel(c2_spatial_proj, c1_spatial_proj)


    def transform(self, X):
        """
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_chs, n_features)
            test samples.

        Returns
        -------
        dv_hdca: ndarray of shape (n_samples, )
            decision values for each sample of X.
        """
        n_test_samples, n_chs, n_features = X.shape

        if (self.n_chs != n_chs) or (self.n_features != n_features):
            raise ValueError('The format of input X is not correct, it should be consistent with that of the training set.')

        # first step: spatial weight vector projection
        test_proj = np.zeros((n_test_samples, self.win_num))
        for idx_win in range(self.win_num):
            win_seq = np.arange(self.win_start_pnt[idx_win], self.win_start_pnt[idx_win]+self.win_len)
            test_data = X[:, :, win_seq]

            test_proj[:, idx_win] = np.matmul(self.spatial_w[idx_win][np.newaxis, ...], test_data).sum(axis=-1).squeeze()

        # second step: time weight vector projection
        dv_hdca = test_proj @ self.time_w.T

        return dv_hdca.squeeze()

    def predict(self, X):
        """
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_chs, n_features)
            test samples.

        Returns
        -------
        y_pred: ndarray of shape (n_samples, )
            Vector containing the class labels for each sample.
        """
        dv_hdca = self.transform(X)
        y_pred = (dv_hdca > self.threshold) * np.ones((40), dtype=int)

        return y_pred


if __name__ == '__main__':

    X = np.random.randn(900, 16, 160)
    y = np.concatenate((np.ones(100), np.ones(100*8)*-1), axis=0)

    hdca = HDCA(fs=200, t_dur=0.1)
    hdca.fit(X, y)

    print('breakpoint')
