import os
import numpy as np
from sleep_stage import print_n_samples_each_class_5,  print_n_samples_each_class_4,  print_n_samples_each_class_3,  print_n_samples_each_class_2
import scipy.io as sio

class NonSeqDataLoader():

    def __init__(self, data_dir, n_folds, fold_idx, n_classes,seq_length):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.n_classes = n_classes
        self.seq_length = seq_length

    def _load_mat_file(self, mat_file):
        """Load data and labels from a mat file."""
        datamat = sio.loadmat(mat_file)
        data = datamat["xx"]
        data_EOG = data
        data_EOG = data_EOG[:, :]   
        data = data_EOG
        if self.n_classes==5:
            labels = np.squeeze(datamat["y_5"])
        elif self.n_classes==4:
            labels = np.squeeze(datamat["y_4"])
        elif self.n_classes==3:
            labels = np.squeeze(datamat["yy"])
        elif self.n_classes==2:
            labels = np.squeeze(datamat["yy"]) 
        labels = labels
        sampling_rate = 128
        return data, labels, sampling_rate
    
    def _load_mat_list_files(self, mat_files):
        """Load data and labels from list of mat files."""
        data = []
        labels = []
        fs = None
        for mat_f in mat_files:
            print ("Loading {} ...".format(mat_f))
            tmp_data, tmp_labels, sampling_rate = self._load_mat_file(mat_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def load_train_data(self, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        matfiles = []
        for idx, f in enumerate(allfiles):
            if ".mat" in f:
                matfiles.append(os.path.join(self.data_dir, f))
        matfiles.sort()

        if n_files is not None:
            matfiles = matfiles[:n_files]

        all_subject_files = []
        temp_f = []
        for idx, f in enumerate(allfiles):     
            if (idx+1) % self.n_folds == (self.fold_idx+1):
                temp_f.append(f)
                all_subject_files.append(f)          
            else:
                all_subject_files.append(f)
        
        subject_files = temp_f
        subject_files_dir = []
        
        for idx, f in enumerate(subject_files):
            subject_files_dir.append(os.path.join(self.data_dir, f))
        subject_files_dir.sort()
        
        train_files = list(set(matfiles)-set(subject_files_dir))
        train_files.sort()

        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        data_train, label_train = self._load_mat_list_files(mat_files=train_files)
        print (" ")
        print ("Load validation set:")
        data_val, label_val = self._load_mat_list_files(mat_files=subject_files_dir)
        print (" ")

        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        # data_train = data_train[:, :, np.newaxis, np.newaxis]
        # data_val = data_val[:, :, np.newaxis, np.newaxis]
        # data_train = data_train[:, :, :, np.newaxis]
        # data_val = data_val[:, :, :, np.newaxis]

        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)
        
        train_epoch, epoch_len = data_train.shape
        val_epoch, _2 = data_val.shape
        
        temp_data_train = np.zeros((len(label_train) - self.seq_length + 1,self.seq_length, epoch_len))#channels_num))
        temp_label_train = np.zeros((len(label_train) - self.seq_length + 1,self.seq_length))
        temp_data_val = np.zeros((len(label_val) - self.seq_length + 1,self.seq_length, epoch_len))#channels_num))
        temp_label_val = np.zeros((len(label_val) - self.seq_length + 1,self.seq_length))
        for len1 in range(len(label_train)-self.seq_length+1):
            temp_data_train[len1,:,:] = data_train[len1:(len1 + self.seq_length),:]
            temp_label_train[len1,:] = label_train[len1:(len1 + self.seq_length)]
        temp_data_train = temp_data_train[:,:,:,np.newaxis]
        
        for len2 in range(len(label_val)-self.seq_length+1):
            temp_data_val[len2,:,:] = data_val[len2:(len2 + self.seq_length),:]
            temp_label_val[len2,:] = label_val[len2:(len2 + self.seq_length)]
        temp_data_val = temp_data_val[:,:,:,np.newaxis]
        
        data_train = temp_data_train
        label_train = temp_label_train
        data_val = temp_data_val
        label_val = temp_label_val

        print ("Training set: {}, {}".format(data_train.shape, label_train.shape))
        if self.n_classes==5:
            print_n_samples_each_class_5(np.hstack(label_train))
        elif self.n_classes==4:
            print_n_samples_each_class_4(np.hstack(label_train))
        elif self.n_classes==3:
            print_n_samples_each_class_3(np.hstack(label_train))
        elif self.n_classes==2:
            print_n_samples_each_class_2(np.hstack(label_train)) 
        print (" ")
        print ("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        if self.n_classes==5:
            print_n_samples_each_class_5(np.hstack(label_val))
        elif self.n_classes==4:
            print_n_samples_each_class_4(np.hstack(label_val))
        elif self.n_classes==3:
            print_n_samples_each_class_3(np.hstack(label_val))
        elif self.n_classes==2:
            print_n_samples_each_class_2(np.hstack(label_val)) 
        print (" ")
        x_train=data_train
        y_train=label_train        
        if self.n_classes==5:
            print_n_samples_each_class_5(np.hstack(y_train))
        elif self.n_classes==4:
            print_n_samples_each_class_4(np.hstack(y_train))
        elif self.n_classes==3:
            print_n_samples_each_class_3(np.hstack(y_train))
        elif self.n_classes==2:
            print_n_samples_each_class_2(np.hstack(y_train)) 
        print (" ")

        return x_train, y_train, data_val, label_val