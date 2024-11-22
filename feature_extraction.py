# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:31:34 2023

@author: User01
"""

import numpy as np
from scipy import signal
import scipy.io as sio
import os
import glob

fs = 128

class feature_extraction():
    def __init__(self, data, fs):
        self.data = data
        self.fs = fs
        
    def psd_ext(self):
    
        eeg = data.reshape(-1)
        print(eeg.shape)
        # 设置FFT窗口大小和重叠大小
        nperseg = 256
        noverlap = nperseg // 2
        
        # 计算EEG信号的功率谱密度
        freqs, psd = signal.welch(eeg, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # 将功率谱密度转换为对数尺度
        log_psd = 10 * np.log10(psd)
        
        # 计算频段内的特征分布
        delta_psd = np.mean(log_psd[(freqs >= 0.5) & (freqs < 4)])
        theta_psd = np.mean(log_psd[(freqs >= 4) & (freqs < 8)])
        alpha_psd = np.mean(log_psd[(freqs >= 8) & (freqs < 13)])
        beta_psd = np.mean(log_psd[(freqs >= 13) & (freqs < 30)])
        gamma_psd = np.mean(log_psd[(freqs >= 30) & (freqs < 50)])
        
        psd_feature = [delta_psd, theta_psd, alpha_psd, beta_psd, gamma_psd]
    
        return psd_feature
    
    def pwr_ext(self):
        
        eeg_data = data.reshape(-1)
        delta_band = [0.5, 4]
        theta_band = [4, 8]
        alpha_band = [8, 13]
        beta_band = [13, 30]
        gamma_band = [30, 50]
        
        # 计算能量
        nyq = 0.5 * fs
        delta_low = delta_band[0] / nyq
        delta_high = delta_band[1] / nyq
        theta_low = theta_band[0] / nyq
        theta_high = theta_band[1] / nyq
        alpha_low = alpha_band[0] / nyq
        alpha_high = alpha_band[1] / nyq
        beta_low = beta_band[0] / nyq
        beta_high = beta_band[1] / nyq
        gamma_low = gamma_band[0] / nyq
        gamma_high = gamma_band[1] / nyq
        
        order = 5
        b, a = signal.butter(order, [delta_low, delta_high], btype='band')
        delta_power = np.sum(np.square(signal.filtfilt(b, a, eeg_data)))
        
        b, a = signal.butter(order, [theta_low, theta_high], btype='band')
        theta_power = np.sum(np.square(signal.filtfilt(b, a, eeg_data)))
        
        b, a = signal.butter(order, [alpha_low, alpha_high], btype='band')
        alpha_power = np.sum(np.square(signal.filtfilt(b, a, eeg_data)))
        
        b, a = signal.butter(order, [beta_low, beta_high], btype='band')
        beta_power = np.sum(np.square(signal.filtfilt(b, a, eeg_data)))
        
        b, a = signal.butter(order, [gamma_low, gamma_high], btype='band')
        gamma_power = np.sum(np.square(signal.filtfilt(b, a, eeg_data)))
        
        pwr_feature = [delta_power, theta_power, alpha_power, beta_power, gamma_power]
        
        return pwr_feature
    
    def sw_fw_ext(self):
        
        eeg = data.reshape(-1)

        sw_band = [1, 4]
        fw_band = [8, 13]
        
        # 计算EEG信号的带通滤波信号
        sw_band_filter = signal.firwin(101, sw_band, pass_zero=False, fs=fs)
        fw_band_filter = signal.firwin(101, fw_band, pass_zero=False, fs=fs)
        
        sw_eeg = signal.filtfilt(sw_band_filter, 1, eeg)
        fw_eeg = signal.filtfilt(fw_band_filter, 1, eeg)
        
        # 计算EEG信号的快波和慢波比例
        sw_power = np.sum(sw_eeg**2) / len(sw_eeg)
        fw_power = np.sum(fw_eeg**2) / len(fw_eeg)
        sw_fw_ratio = sw_power / fw_power

        # 计算EEG信号的相干特性
        freqs, psd = signal.welch(eeg, fs=fs)
        coh = signal.coherence(sw_eeg, fw_eeg, fs=fs)
        
        sw_fw_feature = [sw_power, fw_power, sw_fw_ratio]
        
        return sw_fw_feature

path = 'SC' # rawfile -> reload as mat -> path 
matfile = glob.glob(os.path.join(path, "*.mat"))

allfeature = []
for i in range(len(matfile)):
    mat = sio.loadmat(matfile[i])
    data = mat['xx']
    feature_data = feature_extraction(data, fs)
    psd_feature = feature_data.psd_ext()
    pwr_feature = feature_data.pwr_ext()
    sw_fw_feature = feature_data.sw_fw_ext()
    allfeature.append(psd_feature + pwr_feature + sw_fw_feature)

allfeature = np.array(allfeature)
sio.savemat('data/SC/feature2.mat', {'feature': allfeature})