# -*- coding:utf-8 -*-
'''
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2022.07
Refer:
    [1] Sajda, Paul, et al. "In a blink of an eye and a switch of a transistor: cortically coupled computer vision."
        Proceedings of the IEEE 98.3 (2010): 462-478.
    [2] Marathe et al. "Sliding HDCA: single-trial EEG classification to overcome and quantify temporal variability."
        IEEE Transactions on Neural Systems and Rehabilitation Engineering 22.2 (2014): 201-211.

Application: Binary problems, e.g., single trial recognition for P300-based BCI.

'''

import warnings

import numpy as np
from numpy import ndarray
from scipy import linalg as LA
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from .STDA import LDA_kernel
from .HDCA import HDCA


class sHDCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Sliding Hierarchical Discriminant Component Analysis (sHDCA)

    Parameters
    ----------
    fs: int, unit: Hz.
        sampling rate for the input data.

    t_cut1: float, unit: second.
        the starting time of the whole time window that is cut to extract features and recognize.
        The 0 refers to stimulus onset.
        recommended value = 0.1 seconds in the paper.
        
    t_cut2: float, unit: second.
        the starting time of the whole time window that is cut to extract features and recognize.
        The 0 refers to stimulus onset.
        recommended value = 1.6 seconds in the paper.

    t_used_min: float, unit: second.
        default = 0.3 seconds in the paper.

    t_used_max: float, unit: second.
        default = 0.8 seconds in the paper.

    t_dur: float, unit: second.
        aks. time slices, default = 0.05 seconds in the paper.

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

    Notes
    -----
    Notably, The corresponding time of the first point of input X is t_cut1.

    """

    def __init__(self, fs: int, t_cut1: float, t_cut2: float, t_used_min=0.3, t_used_max=0.8, t_dur=0.05):

        if (t_cut1 > t_used_min) or (t_cut2 < t_used_max):
            raise ValueError('The time setting has some problems, please check the magnitude '
                             'between the values of t_cut1, t_cut2, t_used_min, and t_used_max')

        self.fs = int(fs)
        self.t_cut1 = t_cut1
        self.t_cut2 = t_cut2
        self.t_used_min = t_used_min
        self.t_used_max = t_used_max
        self.t_dur = t_dur


    def fit(self, X, y):
        """Fit sHDCA model according to the given training data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_chs, n_features)
           Training data.

        y : array-like of shape (n_samples,)
            Target values, {-1, 1} or {0, 1}.

        """
        win_hdca_len = self.t_used_max - self.t_used_min
        n_used_feats = int((self.t_cut2 - self.t_cut1 - win_hdca_len) * self.fs)

        n_features = X.shape[-1]
        if int((self.t_cut2 - self.t_cut1)*self.fs) > n_features:
            raise ValueError('The reserved feature length is too short to allow for sliding operations. '
                             'The length should be not less than int((self.t_cut2 - self.t_cut1 + win_len)*self.fs)')

        # format: (n_samples, n_chs, n_features), or (n_samples, )
        X_hdca, X_lr, y_hdca, y_lr = train_test_split(X, y, test_size=0.5, random_state=66, stratify=y)
        loc_pnt_start, loc_pnt_end = int((self.t_used_min - self.t_cut1) * self.fs), int((self.t_used_max - self.t_cut1) * self.fs)

        clf_hdca = HDCA(fs=self.fs, t_dur=self.t_dur)
        clf_hdca.fit(X_hdca[..., loc_pnt_start:loc_pnt_end], y_hdca)  # get spatial and time filter

        n_pnts_hdca = int(win_hdca_len * self.fs)
        feat_scores = np.zeros((X_lr.shape[0], n_used_feats))
        for idx_pnt in range(n_used_feats):
            time_slices_seq = np.arange(idx_pnt, idx_pnt+n_pnts_hdca)
            feat_scores[:, idx_pnt] = clf_hdca.transform(X_lr[:, :, time_slices_seq])  # (n_samples, n_used_feats)

        # step 4: apply LR
        clf_lr = LogisticRegression(random_state=0)
        clf_lr.fit(feat_scores, y_lr)

        self.n_used_feats = n_used_feats
        self.n_pnts_hdca = n_pnts_hdca
        self.clf_hdca = clf_hdca
        self.clf_lr = clf_lr

    def transform(self, X):
        """
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_chs, n_features)
            test samples.

        Returns
        -------
        dv_shdca: ndarray of shape (n_samples, )
            decision values for each sample of X.
        """
        feat_scores = np.zeros((X.shape[0], self.n_used_feats))
        for idx_pnt in range(self.n_used_feats):
            time_slices_seq = np.arange(idx_pnt, idx_pnt+self.n_pnts_hdca)
            feat_scores[:, idx_pnt] = self.clf_hdca.transform(X[:, :, time_slices_seq])  # (n_samples, n_used_feats)

        # y_proba = self.clf_lr.predict_proba(feat_scores)
        y_proba = self.clf_lr.decision_function(feat_scores)

        return y_proba

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
        feat_scores = np.zeros((X.shape[0], self.n_used_feats))
        for idx_pnt in range(self.n_used_feats):
            time_slices_seq = np.arange(idx_pnt, idx_pnt+self.n_pnts_hdca)
            feat_scores[:, idx_pnt] = self.clf_hdca.transform(X[:, :, time_slices_seq])  # (n_samples, n_used_feats)

        y_pred = self.clf_lr.predict(feat_scores)

        return y_pred


if __name__ == '__main__':
    X = np.random.randn(900, 16, 200+100)
    y = np.concatenate((np.ones(500), np.ones(100*4)*-1), axis=0)

    Xtest = np.random.randn(400, 16, 200+100)

    shdca = sHDCA(fs=200, t_cut1=0.1, t_cut2=1.1, t_used_min=0.3, t_used_max=0.8, t_dur=0.05)
    shdca.fit(X, y)

    y_proba = shdca.transform(Xtest)
    y_pred = shdca.predict(Xtest)

    print('breakpoint')


