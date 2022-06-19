# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/07
# License: MIT License
"""
Discriminal Spatial Patterns.
"""
from typing import Optional, List, Tuple, Dict

import numpy as np
from scipy.linalg import eigh
from numpy import ndarray
from scipy.linalg import solve
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

def robust_pattern(W: ndarray, Cx: ndarray, Cs: ndarray) -> ndarray:
    """Transform spatial filters to spatial patterns based on paper [1]_.

    Parameters
    ----------
    W : ndarray
        Spatial filters, shape (n_channels, n_filters).
    Cx : ndarray
        Covariance matrix of eeg data, shape (n_channels, n_channels).
    Cs : ndarray
        Covariance matrix of source data, shape (n_channels, n_channels).

    Returns
    -------
    A : ndarray
        Spatial patterns, shape (n_channels, n_patterns), each column is a spatial pattern.

    References
    ----------
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging." Neuroimage 87 (2014): 96-110.
    """
    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    A = solve(Cs.T, np.dot(Cx, W).T).T
    return A


def xiang_dsp_kernel(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """dsp algorihtm based on paper[1].

    Parameters
    ----------
    X : ndarray
        EEG data assuming removing mean, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels, shape (n_trials, )
    
    Returns
    -------
    W: ndarray
        filters, shape (n_channels, n_filters)
    D: ndarray
        eigenvalues in descending order
    M: ndarray
        template for all classes, shape (n_channel, n_samples)
    A: ndarray
        spatial patterns, shape (n_channels, n_filters)

    Notes
    -----
    1. the implementation removes regularization on within-class scatter matrix Sw.

    References
    ----------
    [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.    
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    # the number of each label
    n_labels = np.array([np.sum(y==label) for label in labels])
    # average template of all trials
    M = np.mean(X, axis=0)
    # class conditional template
    Ms, Ss =zip(*[
        (np.mean(X[y==label], axis=0), np.sum(np.matmul(X[y==label], np.swapaxes(X[y==label], -1, -2)), axis=0)) for label in labels
        ])
    Ms, Ss = np.stack(Ms), np.stack(Ss)
    # within-class scatter matrix
    Sw = np.sum(Ss - n_labels[:, np.newaxis, np.newaxis]*np.matmul(Ms, np.swapaxes(Ms, -1, -2)), axis=0) 
    Ms = Ms - M
    # between-class scatter matrix
    Sb = np.sum(n_labels[:, np.newaxis, np.newaxis]*np.matmul(Ms, np.swapaxes(Ms, -1, -2)), axis=0)

    D, W = eigh(Sb, Sw)
    ix = np.argsort(D)[::-1] # in descending order
    D, W = D[ix], W[:, ix]
    A = robust_pattern(W, Sb, W.T@Sb@W)

    return W, D, M, A

def xiang_dsp_feature(W: ndarray, M: ndarray, X: ndarray,
        n_components: int = 1) -> ndarray:
    """Return DSP features in paper [1]_.

    Parameters
    ----------
    W : ndarray
        spatial filters from csp_kernel, shape (n_channels, n_filters)
    M: ndarray
        common template for all classes, shape (n_channel, n_samples)
    X : ndarray
        eeg data, shape (n_trials, n_channels, n_samples)
    n_components : int, optional
        the first k components to use, usually even number, by default 1

    Returns
    -------
    ndarray
        features of shape (n_trials, n_components, n_samples)

    Raises
    ------
    ValueError
        n_components should less than half of the number of channels

    Notes
    -----
    1. instead of meaning of filtered signals in paper [1]_., we directly return filtered signals.

    References
    ----------
    .. [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    W, M, X = np.copy(W), np.copy(M), np.copy(X)
    max_components = W.shape[1]
    if n_components > max_components:
        raise ValueError("n_components should less than the number of channels")
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    features = np.matmul(W[:, :n_components].T, X - M)
    return features
