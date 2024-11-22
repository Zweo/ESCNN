# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:24:54 2023

@author: User01
"""

from seqcnn_model import seqcnn
import matlab.engine
import scipy.io as io
import glob
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
import joblib
import pickle

total_cm = np.zeros((5, 5))
path = 'data/SC/'
distance_filepath = 'data/SC/distance.mat'
data = io.loadmat(distance_filepath)
distance = data['distance']
for file in [1]:
    data_dir = path  + 'test/' + file + '/'
    index = 1
    prob_each = []
    d = []
    for t in range(distance.shape[-1]):
        model_dir_t = path  + 'weights/' + file +  + '/fold' + str(
            t) + '/feature_learning/f_model_weights.h5'
        prob_tmp, y_true = seqcnn(model_dir_t, data_dir)
        prob_each.append(prob_tmp)
        d.append(distance[index, t])
    d = np.array(d)
    d = d**2
    d = d.reshape(-1, 1)
    summ = sum(d)
    d0 = 1 / (d / summ)
    prob = prob_tmp * 0
    for ii in range(len(d0)):
        prob += prob_each[ii] * d0[ii]
    y_pred = np.argmax(prob, axis=1)
    y_pred = y_pred.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)
    train_cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print("train: (acc={:.3f},f1={:.3f})".format(acc, f1))
    print(train_cm)
    total_cm += train_cm
print(total_cm)
