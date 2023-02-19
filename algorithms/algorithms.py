# -*- coding:utf-8 -*-
'''
@ author: Jin Han
@ email: jinhan9165@gmail.com
@ Created on: 2020-03-24
version 1.0
update:
Refer: see belows

Application: classic algorithms,
    TRCA: Task-related Component Analysis
        Refer: Nakanishi, et al. "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis."
            IEEE Trans. Biomed. Eng. 65.1 (2017): 104-112.
    CCA: Canonical correlation analysis
        Refer: None
    Extended CCA: Extended Canonical correlation analysis
        Refer: Nakanishi, et al. "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis."
            IEEE Trans. Biomed. Eng. 65.1 (2017): 104-112.
    tt-CCA: Transfer Template CCA
        Refer: Yuan et al. "Enhancing performances of SSVEP-based brain–computer interfaces via exploiting inter-subject
            information." Journal of neural engineering, 12.4 (2015): 046006.
    DSP: Discriminative Spatial Patterns
        Refer: Liao, et al. "Combining spatial filters for the classification of single-trial EEG in a finger movement task."
            IEEE Trans. Biomed. Eng. 54.5 (2007): 821-831.
    DCPM:
'''
import numpy as np
from numpy import linalg as LA
from sklearn.cross_decomposition import CCA

from STDA import LDA_kernel

def trca_compute(Xin, subspace_idx=None):
    '''
    task-related component analysis (TRCA)
    :param Xin: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (or n_trials)) for a certain class.
    :param subspace_idx: int, optional (default=None)
        Selected the first subspaces_idx subspaces of projection computed by TRCA.
        If None, all feature space selected.
    :return:
        eig_vals: 1D ndarray
            (n_channels,)
        eig_vector: 2D ndarray
            (n_channels * n_channels)
    '''
    # print('Now, algorithm TRCA is running...')
    n_chans = Xin.shape[0]
    n_epochs = Xin.shape[2]

    # zero means
    values_mean = Xin.mean(axis=1, keepdims=True)
    # Xin_std = Xin.std(axis=1, ddof=1, keepdims=True)
    Xin = (Xin-values_mean)  # /Xin_std

    S = np.zeros((n_chans, n_chans))
    # computation of correlation matrices
    for epoch_i in range(n_epochs):
        for epoch_j in range(n_epochs):
            Xin_i = Xin[:,:,epoch_i]
            Xin_j = Xin[:,:,epoch_j]
            S += np.dot(Xin_i, Xin_j.T)

    Xin_i2 = Xin.reshape((Xin.shape[0], -1), order='F')
    Xin_i2 = Xin_i2 - Xin_i2.mean(axis=1, keepdims=True)
    Q = np.dot(Xin_i2, Xin_i2.T)

    eig_vals, eig_vectors = LA.eig(np.dot(LA.inv(Q),S)) # or LA.pinv for general

    eig_vectors = eig_vectors[:, eig_vals.argsort()[::-1]] # return indices in ascending order and reverse

    eig_vals.sort()
    eig_vals = eig_vals[::-1] # sort in descending order
    # print('Now, algorithm TRCA is finished.')

    eig_vals = eig_vals if subspace_idx is None else eig_vals[:subspace_idx]
    eig_vectors = eig_vectors if subspace_idx is None else eig_vectors[:,:subspace_idx]

    return eig_vals, eig_vectors

def cca_manu(Xin, Yin):
    '''
    refer: https://www.cnblogs.com/pinard/p/6288716.html
    :param Xin: ndarray
        (n_channels_1 * n_time * n_epochs (i.e. n_trials))
    :param Yin: ndarray
        (n_channels_2 * n_time * n_epochs (i.e. n_trials))
    :return:
        eig_vectors_x: (n_channels1 * n_channels1), ordered by columns.
        eig_vectors_y: (n_channels2 * n_channels2), ordered by columns.
        RR: (n_channels, )
    '''
    if Xin.ndim == 3:
        Xin = Xin.mean(axis=-1, keepdims=False)
        Yin = Yin.mean(axis=-1, keepdims=False)

    # print('Now, algorithm cca_manu is running...')
    Xin -= Xin.mean(axis=1, keepdims=True)
    Yin -= Yin.mean(axis=1, keepdims=True)
    cov_xx = np.cov(Xin, rowvar=True, bias=False)
    cov_yy = np.cov(Yin, rowvar=True, bias=False)
    # cross covariance
    cov_xy = np.dot(Xin, Yin.T)/(Xin.shape[1]-1)
    # cov_yx = cov_xy.T

    # eps = np.finfo(cov_xx.dtype).eps
    # cov_xx = np.maximum(cov_xx, eps)
    # cov_yy = np.maximum(cov_yy, eps)
    # cov_xy = np.maximum(cov_xy, eps)
    # inv_2_xx = LA.inv(cov_xx**0.5)
    # inv_2_yy = LA.inv(cov_yy**0.5)

    U, sigma_xx, VT = LA.svd(cov_xx)
    sigma_xx = sigma_xx**(-0.5)
    inv_2_xx = U.dot(np.diag(sigma_xx)).dot(VT)

    U, sigma_yy, VT = LA.svd(cov_yy)
    sigma_yy = sigma_yy**(-0.5)
    inv_2_yy = U.dot(np.diag(sigma_yy)).dot(VT)

    M = inv_2_xx.dot(cov_xy).dot(inv_2_yy)
    U, sigma_M, VT = LA.svd(M)

    eig_vectors_x = inv_2_xx.dot(U)
    eig_vectors_y = inv_2_yy.dot(VT.T)

    # compute correlation coeficient
    X_proj = np.dot(eig_vectors_x.T, Xin)
    Y_proj = np.dot(eig_vectors_y.T, Yin)
    n_dims = X_proj.shape[0] if X_proj.shape[0] < Y_proj.shape[0] else Y_proj.shape[0]
    rr_coef = np.zeros((n_dims))
    for i in range(n_dims):
        rr_coef[i] = np.corrcoef(X_proj[i,:], Y_proj[i,:])[0,1]
    # print('Now, algorithm cca_manu is finished.')

    return eig_vectors_x, eig_vectors_y, rr_coef

def extended_cca(Xtest, Xtrain, freq_stim, fs, t_begin, t_end, n_hf=5, init_phase=None, subspace=None):
    '''
    Extended CCA, Rcommended!
    :param Xtest: 2D, ndarray.
        n_channnes * n_features
    :param Xtrain: 3D, nadrray.
        (n_channels * n_features * n_epochs) for a certain class.
    :param freq_stim: float, unit: Hz.
        e.g. 13.8
    :param fs: int, unit: Hz.
    :param t_begin: float, unit: second.
        e.g. 0
    :param t_end: float, unit: second.
        e.g. 0.5
    :param n_hf: the num of harmonics frequency, default=5.
    :param init_phase: int, optional(default=None, or 0.)
        initial phase for the fixed stimulation frequency (i.e. freq_stim Hz)
        It is recommended when p-CCA is using.
    :param subspace: int, optional(default=None)
        Selected the first subspaces_idx subspaces of projection computed by CCA.
        If None, the half of n_channels(generally) is choosed.
    :return: rr_fusion: float,
        the fusion correlation coefficient.
    '''
    if t_begin > t_end:
        raise ValueError('t_begin should be less than t_end.')
    if Xtrain.ndim == 3:
        Xtrain = Xtrain.mean(axis=-1, keepdims=False)

    # zero means
    Xtrain -= Xtrain.mean(axis=1, keepdims=True)
    Xtest -= Xtest.mean(axis=1, keepdims=True)

    init_phase = 0 if init_phase is None else init_phase
    t1 = np.ceil(fs*t_begin)
    t2 = np.ceil(fs*t_end)
    t_time = np.arange(t1, t2+1, 1)/fs

    # generate sin-cos template
    Yf = None
    for idx_hf in range(1, n_hf+1):
        sin_tmp = np.sin(2*np.pi*freq_stim*idx_hf*t_time+init_phase)
        cos_tmp = np.cos(2*np.pi*freq_stim*idx_hf*t_time+init_phase)
        Yf_tmp = np.vstack((sin_tmp, cos_tmp)).T
        Yf = Yf_tmp if Yf is None else np.hstack((Yf, Yf_tmp))  # n_time * n_chans

    # recognition by extended CCA
    n_chans, n_features = Xtrain.shape
    n_Yf = Yf.shape[0]
    if subspace is None:
        subspace = int(np.ceil(n_chans/2)) if n_chans <= n_Yf else int(np.ceil(n_Yf/2))

    rr_coef = np.zeros(5)
    Xtest = Xtest.T  # n_time * n_chans
    Xtrain = Xtrain.T  # n_time * n_chans

    u_x, v_y, RR = cca_manu(Xtest.T, Yf.T)
    tmp_1 = Xtest.dot(u_x[:,:subspace]).reshape(1, -1)
    rr_coef[0] = RR[:subspace].mean()
    # tmp_2 = Yf.dot(v_y[:, :subspace]).reshape(1, -1)  # better than RR[:subspace].mean()
    # rr_coef[0] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    tmp_2 = Xtrain.dot(u_x[:,:subspace]).reshape(1, -1)
    rr_coef[2] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    u_x, v_y, _ = cca_manu(Xtest.T, Xtrain.T)
    tmp_1 = Xtest.dot(u_x[:,:subspace]).reshape(1, -1)
    tmp_2 = Xtrain.dot(u_x[:,:subspace]).reshape(1, -1)
    rr_coef[1] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    tmp_1 = Xtrain.dot(u_x[:,:subspace]).reshape(1, -1)
    tmp_2 = Xtrain.dot(v_y[:,:subspace]).reshape(1, -1)
    rr_coef[4] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    u_x, v_y, _ = cca_manu(Xtrain.T, Yf.T)
    tmp_1 = Xtest.dot(u_x[:,:subspace]).reshape(1, -1)
    tmp_2 = Xtrain.dot(u_x[:,:subspace]).reshape(1, -1)
    rr_coef[3] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    rr_fusion = (np.sign(rr_coef)*(rr_coef**2)).sum()

    return rr_fusion

def extended_cca2(Xtest, Xtrain, freq_stim, fs, t_begin, t_end, n_hf=5, init_phase=None, subspce_idx=None):
    '''
    Extended CCA, Using sklearn's CCA, NO Recommended!
    :param Xtest: 2D, ndarray.
        n_channnes * n_features
    :param Xtrain: 3D, nadrray.
        (n_channels * n_features * n_epochs) for a certain class.
    :param freq_stim: float, unit: Hz.
        e.g. 13.8
    :param fs: int, unit: Hz.
    :param t_begin: float, unit: second.
        e.g. 0.14
    :param t_end: float, unit: second.
        e.g. 0.14+0.5
    :param n_hf: the num of harmonics frequency, default=5.
    :param init_phase: int, optional(default=None, or 0.)
        initial phase for the fixed stimulation frequency (i.e. freq_stim Hz)
        It is recommended when p-CCA is using.
    :param subspce_idx: int, optional(default=None)
        Selected the first subspaces_idx subspaces of projection computed by CCA.
        If None, the half of n_components(=n_channels, generally) is choosed.
    :return:
    '''
    if t_begin > t_end:
        raise ValueError('t_begin should be less than t_end.')
    if Xtrain.ndim == 3:
        Xtrain = Xtrain.mean(axis=-1, keepdims=False)

    # zero means
    Xtrain -= Xtrain.mean(axis=1, keepdims=True)
    Xtest -= Xtest.mean(axis=1, keepdims=True)

    init_phase = 0 if init_phase is None else init_phase
    t1 = np.ceil(fs*t_begin)
    t2 = np.ceil(fs*t_end)
    t_time = np.arange(t1, t2+1, 1)/fs

    # generate sin-cos template
    Yf = None
    for idx_hf in range(1, n_hf+1):
        sin_tmp = np.sin(2*np.pi*freq_stim*idx_hf*t_time+init_phase)
        cos_tmp = np.cos(2*np.pi*freq_stim*idx_hf*t_time+init_phase)
        Yf_tmp = np.vstack((sin_tmp, cos_tmp)).T
        Yf = Yf_tmp if Yf is None else np.hstack((Yf, Yf_tmp))

    # recognition by extended CCA
    n_chans, n_features = Xtrain.shape
    n_Yf = Yf.shape[1]
    n_components = int(np.ceil(n_chans/2)) if n_chans <= n_Yf else int(np.ceil(n_Yf/2))
    Xtrain = Xtrain.T  # n_features * n_chans, i.e. n_samples * n_features
    Xtest = Xtest.T

    rr_coef = np.zeros(5)
    clf_cca = CCA(n_components=n_components, scale=False)

    clf_cca.fit(Xtest, Yf)
    tmp_1 = Xtest.dot(clf_cca.x_weights_[:,:subspce_idx]).reshape(1, -1)
    tmp_2 = Yf.dot(clf_cca.y_weights_[:, :subspce_idx]).reshape(1, -1)
    rr_coef[0] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    tmp_2 = Xtrain.dot(clf_cca.x_weights_[:,:subspce_idx]).reshape(1, -1)
    rr_coef[2] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    clf_cca = CCA(n_components=n_components, scale=False)
    clf_cca.fit(Xtest, Xtrain)
    tmp_1 = Xtest.dot(clf_cca.x_weights_[:,:subspce_idx]).reshape(1, -1)
    tmp_2 = Xtrain.dot(clf_cca.x_weights_[:,:subspce_idx]).reshape(1, -1)
    rr_coef[1] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    tmp_1 = Xtrain.dot(clf_cca.x_weights_[:,:subspce_idx]).reshape(1, -1)
    tmp_2 = Xtrain.dot(clf_cca.y_weights_[:,:subspce_idx]).reshape(1, -1)
    rr_coef[4] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    clf_cca = CCA(n_components=n_components, scale=False)
    clf_cca.fit(Xtrain, Yf)
    tmp_1 = Xtest.dot(clf_cca.x_weights_[:,:subspce_idx]).reshape(1, -1)
    tmp_2 = Xtrain.dot(clf_cca.x_weights_[:,:subspce_idx]).reshape(1, -1)
    rr_coef[3] = np.corrcoef(tmp_1, tmp_2)[0, 1]

    rr_fusion = (np.sign(rr_coef)*(rr_coef**2)).sum()

    return rr_fusion

def standard_cca(Xtest, freq_stim, fs, t_begin, t_end, n_hf=5, init_phase=None, subspace=None):
    '''
    Standard CCA for train-free classification of SSVEP, i.e. only with reference signal and test signal.
    :param Xtest: 2D, ndarray.
        n_channnes * n_features
    :param Xtrain: 3D, nadrray.
        (n_channels * n_features * n_epochs) for a certain class.
    :param freq_stim: float, unit: Hz.
        e.g. 13.8
    :param fs: int, unit: Hz.
    :param t_begin: float, unit: second.
        e.g. 0
    :param t_end: float, unit: second.
        e.g. 0.5
    :param n_hf: the num of harmonics frequency, default=5.
    :param init_phase: int, optional(default=None, or 0.)
        initial phase for the fixed stimulation frequency (i.e. freq_stim Hz)
        It is recommended when p-CCA is using.
    :param subspace: int, optional(default=None)
        Selected the first subspaces_idx subspaces of projection computed by CCA.
        If None, the half of n_channels(generally) is choosed.
    :return:
    '''
    if t_begin > t_end:
        raise ValueError('t_begin should be less than t_end.')
    if Xtest.ndim == 3:
        raise ValueError('Xtest should be 2-D array instead of 3D')

    # zero means
    Xtest -= Xtest.mean(axis=1, keepdims=True)

    init_phase = 0 if init_phase is None else init_phase
    t1 = np.ceil(fs*t_begin)
    t2 = np.ceil(fs*t_end)
    t_time = np.arange(t1, t2+1, 1)/fs

    # generate sin-cos template
    Yf = None
    for idx_hf in range(1, n_hf+1):
        sin_tmp = np.sin(2*np.pi*freq_stim*idx_hf*t_time+init_phase)
        cos_tmp = np.cos(2*np.pi*freq_stim*idx_hf*t_time+init_phase)
        Yf_tmp = np.vstack((sin_tmp, cos_tmp)).T
        Yf = Yf_tmp if Yf is None else np.hstack((Yf, Yf_tmp))  # n_time * n_chans

    # recognition by extended CCA
    n_chans, n_features = Xtest.shape
    n_Yf = Yf.shape[0]
    if subspace is None:
        subspace = int(np.ceil(n_chans/2)) if n_chans <= n_Yf else int(np.ceil(n_Yf/2))

    eig_vectors_x, eig_vectors_y, rr_coef = cca_manu(Xtest, Yf.T)

    return eig_vectors_x[:, :subspace], eig_vectors_y[:, :subspace], rr_coef[:subspace].mean()



def tt_cca():
    pass

def dsp_compute(Xclass1, Xclass2):
    '''
    Discriminative Spatial Patterns (DSP)
    :param Xclass1: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :param Xclass2: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :return:
        eig_vals: ndarray
            (n_channels,)
        eig_vector: ndarray
            (n_channels * n_channels)
    '''
    if Xclass1.shape[0] != Xclass2.shape[0] or Xclass1.shape[1] != Xclass2.shape[1]:
        raise ValueError('The selected channels or sample points of two classes should be SAME.')

    # print('Now, algorithm DSP is running...')
    template_1 = Xclass1.mean(axis=2, keepdims=False)
    template_2 = Xclass2.mean(axis=2, keepdims=False)

    template_1 -= template_1.mean(axis=1, keepdims=True)
    template_2 -= template_2.mean(axis=1, keepdims=True)
    X_buff = np.vstack((template_1, template_2))

    cov_all = np.cov(X_buff, rowvar=True, bias=False)
    cov11 = cov_all[:template_1.shape[0], :template_1.shape[0]] # covariance of variables in Class1
    cov22 = cov_all[template_1.shape[0]:, template_1.shape[0]:] # covariance of variables in Class2
    cov12 = cov_all[:template_1.shape[0], template_1.shape[0]:] # covariance of variables between Class1 and Class2
    cov21 = cov_all[template_1.shape[0]:, :template_1.shape[0]] # covariance of variables between Class2 and Class1
    S_B = cov11 + cov22 - cov12 - cov21 # the between-class scatter matrix

    covB = np.empty((Xclass1.shape[0], Xclass1.shape[0], Xclass1.shape[2]))
    for i in range(Xclass1.shape[2]):
        covB[:,:,i] = np.cov((Xclass1[:,:,i]-template_1))
    cov_b1 = covB.mean(axis=-1, keepdims=False)

    for i in range(Xclass2.shape[2]):
        covB[:,:,i] = np.cov((Xclass2[:,:,i]-template_2))
    cov_b2 = covB.mean(axis=-1, keepdims=False)
    S_W = cov_b1 + cov_b2

    eig_vals, eig_vectors = LA.eig(np.dot(LA.inv(S_W),S_B))
    eig_vectors = eig_vectors[:, eig_vals.argsort()[::-1]] # return indices in ascending order and reverse

    eig_vals.sort()
    eig_vals = eig_vals[::-1] # sort in descending order
    # print('Now, algorithm DSP is finished.')

    return eig_vals, eig_vectors

def dcpm_compute(Xclass1, Xclass2, Xtest, dsp_idx=None, cca_idx=None, cca_rr_idx=None):
    '''
    :param Xclass1: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :param Xclass2: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :param Xtest: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :param dsp_idx: int
        extract first dsp_idx columns for DSP. If None, set default values (half).
    :param cca_idx: int
        extract first cca_idx columns for CCA. If None, set default values (half).
    :param cca_rr_idx: int
        extract first cca_rr_idx values for coefficient of CCA. If None, set default values (half).
    :return:
    '''
    if Xtest.ndim < 2:
        raise ValueError('Xtest should be equal to or greater than two dimensions.')

    # print('Now, algorithm DCPM is running...')
    dsp_vals, dsp_vectors = dsp_compute(Xclass1, Xclass2)

    dsp_idx = int(dsp_idx) if dsp_idx is not None else int(np.round(Xclass1.shape[0]/2))
    if Xclass1.ndim == 3:
        Xclass1 = Xclass1.mean(axis=-1)
    if Xclass2.ndim == 3:
        Xclass2 = Xclass2.mean(axis=-1)
    Xclass1 = Xclass1 - Xclass1.mean(axis=-1, keepdims=True)
    Xclass2 = Xclass2 - Xclass2.mean(axis=-1, keepdims=True)

    # average points across trials projected on feature subspaces of DSP
    template_1 = np.dot(dsp_vectors[:,:dsp_idx].T, Xclass1) # out: (dsp_idx * n_times)
    template_2 = np.dot(dsp_vectors[:,:dsp_idx].T, Xclass2) # out: (dsp_idx * n_times)

    # zero means
    if Xtest.ndim == 3:
        dim_idx = Xtest.shape[-1]
        Xtest = Xtest - Xtest.mean(axis=1, keepdims=True)
    else:
        dim_idx = 1
        Xtest = Xtest - Xtest.mean(axis=-1, keepdims=True)

    rr_coef = np.zeros((2,5,dim_idx))
    for i in range(dim_idx):
        test_data = np.dot(dsp_vectors[:,:dsp_idx].T, Xtest[:,:,i]) # out: (dsp_idx * n_times)

        rr_coef[0,0,i] = np.corrcoef(template_1.reshape((1,-1)), test_data.reshape((1,-1)))[0,1] # p11
        rr_coef[1,0,i] = np.corrcoef(template_2.reshape((1,-1)), test_data.reshape((1,-1)))[0,1] # p21
        rr_coef[0,1,i] = -np.cov(template_1-test_data).diagonal().mean() # p12
        rr_coef[1,1,i] = -np.cov(template_2-test_data).diagonal().mean() # p22

        U_class1, V_test1, rr1 = cca_manu(template_1, test_data)
        U_class2, V_test2, rr2 = cca_manu(template_2, test_data)

        cca_rr_idx = int(cca_rr_idx) if cca_rr_idx is not None else int(np.round(rr1.shape[-1]/2))
        cca_idx = int(cca_idx) if cca_idx is not None else int(np.round(U_class1.shape[-1]/2))
        rr_coef[0,2,i] = rr1[:cca_rr_idx].mean() # p13
        rr_coef[1,2,i] = rr2[:cca_rr_idx].mean() # p23

        buff1 = U_class1[:, :cca_idx].T.dot(template_1).reshape((1,-1))
        buff2 = U_class1[:, :cca_idx].T.dot(test_data).reshape((1,-1))
        rr_coef[0,3,i] = np.corrcoef(buff1, buff2)[0,1] # p14

        buff1 = U_class2[:, :cca_idx].T.dot(template_2).reshape((1,-1))
        buff2 = U_class2[:, :cca_idx].T.dot(test_data).reshape((1,-1))
        rr_coef[1,3,i] = np.corrcoef(buff1, buff2)[0,1] # p24

        buff1 = V_test1[:, :cca_idx].T.dot(template_1).reshape((1,-1))
        buff2 = V_test1[:, :cca_idx].T.dot(test_data).reshape((1,-1))
        rr_coef[0,4,i] = np.corrcoef(buff1, buff2)[0,1] # p15

        buff1 = V_test2[:, :cca_idx].T.dot(template_2).reshape((1,-1))
        buff2 = V_test2[:, :cca_idx].T.dot(test_data).reshape((1,-1))
        rr_coef[1,4,i] = np.corrcoef(buff1, buff2)[0,1] # p25

    idx_using = [0, 1]  # 0, 1, 4
    rr_ = rr_coef[:, idx_using, :].sum(axis=1, keepdims=False)

    return rr_[0, :] - rr_[1,:]

def dsp_lda(Xclass1, Xclass2, Xtest, dsp_idx=1):
    '''
    :param Xclass1: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :param Xclass2: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :param Xtest: ndarray
        (n_channels * num of sample points (i.e. n_times) * n_epochs (i.e. n_trials))
    :param dsp_idx: int
        extract first dsp_idx columns for DSP. If None, set default values (half).
    :param cca_idx: int
        extract first cca_idx columns for CCA. If None, set default values (half).
    :param cca_rr_idx: int
        extract first cca_rr_idx values for coefficient of CCA. If None, set default values (half).
    :return:
    '''
    if Xtest.ndim < 2:
        raise ValueError('Xtest should be equal to or greater than two dimensions.')

    # print('Now, algorithm DCPM is running...')
    dsp_vals, dsp_vectors = dsp_compute(Xclass1, Xclass2)

    dsp_idx = int(dsp_idx) if dsp_idx is not None else int(np.round(Xclass1.shape[0]/2))
    # if Xclass1.ndim == 3:
    #     Xclass_tmp1 = Xclass1.mean(axis=-1)
    # if Xclass2.ndim == 3:
    #     Xclass_tmp2 = Xclass2.mean(axis=-1)
    # Xclass_tmp1 = Xclass_tmp1 - Xclass_tmp1.mean(axis=-1, keepdims=True)
    # Xclass_tmp2 = Xclass_tmp2 - Xclass_tmp2.mean(axis=-1, keepdims=True)

    # average points across trials projected on feature subspaces of DSP
    tmp_c1 = np.matmul(dsp_vectors[:,:dsp_idx][np.newaxis,:].transpose([0,2,1]), Xclass1.transpose([2, 0, 1])).squeeze() # out: (dsp_idx * n_times)
    tmp_c2 = np.matmul(dsp_vectors[:,:dsp_idx][np.newaxis,:].transpose([0,2,1]), Xclass2.transpose([2, 0, 1])).squeeze() # out: (dsp_idx * n_times)

    tmp_test = np.matmul(dsp_vectors[:,:dsp_idx][np.newaxis,:].transpose([0,2,1]), Xtest.transpose([2, 0, 1])).squeeze() # out: (dsp_idx * n_times)

    weight_vec, lda_threshold = LDA_kernel(tmp_c1, tmp_c2)

    dv_proba = weight_vec @ tmp_test.T

    return dv_proba



if __name__ == '__main__':
    # unit tests
    # for TRCA
    # Xin = np.random.randn(9,250,6)
    # eig_vals, eig_vectors = trca_compute(Xin)
    #
    # # for DSP
    # Xclass1 = np.random.randint(0,100,(9,125,6))
    # Xclass2 = np.random.randint(0,100,(9,125,6))
    # dsp_compute(Xclass1, Xclass2)
    #
    # # for cca_manu
    # Xin = np.random.randint(0,100,(3,6))
    # Yin = np.random.randint(0,100,(3,6))
    # eig_vectors_x, eig_vectors_y, RR = cca_manu(Xin, Yin)

    # for Extended CCA
    Xtest = np.random.rand(15, 251)
    Xtrain =  np.random.rand(15, 251, 5)
    rr = extended_cca(Xtest, Xtrain, 15, 250, 0, 1, 5, init_phase=None, subspace=None)


    # for DCPM
    Xclass1 = np.random.randint(0,100,(9,125,6))
    Xclass2 = np.random.randint(0,100,(9,125,6))
    Xtest = np.random.randint(0,100,(9,125,6))
    dcpm_compute(Xclass1, Xclass2, Xtest)

    print('breakpoint')
