import os
import tensorflow as tf
import numpy as np
import time
# import tf_slim
from datetime import datetime
from tensorflow.keras import layers, models, Input, regularizers
from utils.data_loader_multich2 import NonSeqDataLoader, SeqDataLoader
from utils.utils import iterate_minibatches, iterate_batch_seq_minibatches
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Model

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices:
#  tf.config.experimental.set_memory_growth(physical_devices[0], True8 )
'''
tf.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
from keras import backend as K
K.clear_session()
'''

time_start = time.time()  # 记录开始时间

n_folds=2
fold_idx=-1
resume=False
n_classes=5
EPOCH_SEC_LEN=30
SAMPLING_RATE=128
seq_length=30

batch_size = 32
input_dims = SAMPLING_RATE*EPOCH_SEC_LEN

def print_performance(n_train_examples, n_valid_examples, train_cm, valid_cm, 
                      epoch, n_epochs, train_duration, train_loss, train_acc, 
                      train_f1, valid_duration, valid_loss, valid_acc, valid_f1):
    # Print performance
    if ((epoch) % 10 == 0) or ((epoch) == n_epochs):
        print (" ")
        print ("[{}] epoch {}:".format(datetime.now(), epoch))
        print ("train ({:.3f} sec): n={}, loss={:.3f}, acc={:.3f}, "
               "f1={:.3f}".format(train_duration, n_train_examples,
                train_loss, train_acc, train_f1))
        print (train_cm)
        print ("valid ({:.3f} sec): n={}, loss={:.3f}, acc={:.3f}, "
            "f1={:.3f}".format(valid_duration, n_valid_examples,
                valid_loss, valid_acc, valid_f1))
        print (valid_cm)
        print (" ")
    else:
        print ("epoch {}: train ({:.2f} sec): n={}, loss={:.3f}, "
            "acc={:.3f}, f1={:.3f} | "
            "valid ({:.2f} sec): n={}, loss={:.3f}, "
            "acc={:.3f}, f1={:.3f}".format(epoch, train_duration, 
                 n_train_examples, train_loss, train_acc, train_f1, valid_duration, 
                 n_valid_examples, valid_loss, valid_acc, valid_f1))
                
        
class Conv1D_Block(tf.keras.Model):
  def __init__(self, filter_size, n_filters, stride, wd=0):
    super(Conv1D_Block, self).__init__(name='')
    
    self.conv_1d = layers.Conv2D(n_filters, (2, filter_size), (1, stride),  
                                 padding='SAME', kernel_regularizer=regularizers.l2(wd))
    self.bn = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)
   
  def call(self, input_tensor, training=False):
    x = self.conv_1d(input_tensor)
    x = self.bn(x, training=training)
    x = tf.nn.relu(x)
    
    return x

class Dense_Block(tf.keras.Model):
  def __init__(self, hidden_size):
    super(Dense_Block, self).__init__(name='')
    
    self.Dense = layers.Dense(hidden_size)
    self.bn = layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)
   
  def call(self, input_tensor, training=False):
    x = self.Dense(input_tensor)
    x = self.bn(x, training=training)
    x = tf.nn.relu(x)
    
    return x

class SELayer(tf.keras.Model):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 =  layers.Dense(channel//reduction, activation = None)
        self.fc3 =  layers.Dense(channel, activation = None)

    def call(self, x):
        shape1 = x.shape
        # print(shape1)
        _,b,c,d = shape1.as_list()
        # b,c = shape1.as_list()
        y = tf.reduce_mean(x,[1,2])
        # y = x  #合并后加se用这一行，反之屏蔽
        # print(y.shape)
        y = self.fc1(y)
        y = tf.nn.relu(y)
        y = self.fc3(y)
        y = tf.sigmoid(y)
        y = tf.reshape(y,[-1,1,1,d])
        # y = tf.reshape(y,[-1,c])
        return x * y


def feature_learning_model(input_dims, n_classes):
    x = Input(shape=(seq_length, input_dims, 1), dtype='float32', name='Input')
    ######### CNNs with detail filter size ######### 
    x_1 = Conv1D_Block(64, 128, 6, 1e-3)(x)
    x_1 = layers.MaxPooling2D((1, 8),(1, 8))(x_1)
    x_1 = layers.Dropout(0.5)(x_1)
    x_1 = Conv1D_Block(6, 128, 1)(x_1)
    x_1 = Conv1D_Block(6, 128, 1)(x_1)
    x_1 = Conv1D_Block(6, 128, 1)(x_1)
    x_1 = Conv1D_Block(6, 128, 1)(x_1)
    x_1 = layers.MaxPooling2D((1, 4),(1, 4))(x_1)
    # print(x_1.shape)
    # print("###")
    x_1 = SELayer(128,reduction=16)(x_1) 
    shape1 = x_1.get_shape()
    x_1 = tf.reshape(x_1, [-1, shape1[1] , shape1[2] * shape1[3]])  
              
    ######### CNNs with shape filter size #########    
    x_2 = Conv1D_Block(640, 128, 64, 1e-3)(x)
    x_2 = layers.MaxPooling2D((1, 6),(1, 6))(x_2)
    x_2 = layers.Dropout(0.5)(x_2)
    x_2 = Conv1D_Block(10, 128, 1)(x_2)
    x_2 = Conv1D_Block(10, 128, 1)(x_2)
    x_2 = Conv1D_Block(10, 128, 1)(x_2)
    x_2 = Conv1D_Block(10, 128, 1)(x_2)
    x_2 = layers.MaxPooling2D((1, 2),(1, 2))(x_2)
    x_2 = SELayer(128,reduction=16)(x_2)  
    shape2 = x_2.get_shape()
    x_2 = tf.reshape(x_2, [-1, shape2[1] , shape2[2] * shape2[3]])     
      
    
    
    ######### concatenate and filter ######### 
    x_c = layers.concatenate([x_1, x_2], axis=2)
    x_c = layers.Dropout(0.5)(x_c)
    x_c = Dense_Block(800)(x_c)
    x_c = layers.Dropout(0.5)(x_c)
    x_c = Dense_Block(400)(x_c)
    # x_c = SELayer(400, reduction=16)(x_c)
    model = models.Model(x, x_c)
    return model

f_1_model = feature_learning_model(input_dims, n_classes)
# f_1_model.summary()

def feature_learning_lastdense(n_classes):
    x = Input(shape=(seq_length, 400), dtype='float32', name='Input')
    output = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(x, output)
    return model
    
f_2_model = feature_learning_lastdense(n_classes)
# f_2_model.summary()

f_model = models.Sequential()   
f_model.add(f_1_model)
f_model.add(f_2_model)

# 
###########################################################    



##########################################################

@tf.function
def f_train_step(inputs, labels, training=True):
    with tf.GradientTape() as tape:
        pred = f_model(inputs, training=training)
        regular_loss = tf.math.add_n(f_model.losses)
        pred_loss = loss_fn(labels, pred)
        total_loss = pred_loss + regular_loss

    gradients = tape.gradient(total_loss, f_model.trainable_variables)
    optimizer_f.apply_gradients(zip(gradients, f_model.trainable_variables))
    
    pred_y = tf.argmax(pred, axis=2)
    return pred_y, total_loss, pred

def f_valid_step(inputs, labels, training=False):
    pred = f_model(inputs, training=training)
    regular_loss = tf.math.add_n(f_model.losses)
    pred_loss = loss_fn(labels, pred)
    total_loss = pred_loss + regular_loss
    pred_y = tf.argmax(pred, axis=2)
    return pred_y, total_loss, pred 

def f_run_epoch(inputs, targets, batch_size, training=True):
    start_time = time.time()
    y = []
    y_prob = []
    y_true = []
    total_loss, n_batches = 0.0, 0
    for x_batch, y_batch in iterate_minibatches(inputs, targets, 
                                                batch_size, shuffle=False):
        _,seq_length,input_dims,dims = x_batch.shape
        # x_batch = x_batch.reshape(batch_size*seq_length,input_dims,dims)
        # x_batch = x_batch[:,:,:,np.newaxis]
        # y_batch = y_batch.reshape(batch_size*seq_length)
        if training == True:
            y_pred, loss_value, y_predprobility = f_train_step(x_batch, y_batch, training=training)           
        else:
            y_pred, loss_value, y_predprobility = f_valid_step(x_batch, y_batch, training=training)     
        
        total_loss += loss_value
        n_batches += 1
        y_pred = tf.reshape(y_pred, [batch_size, seq_length])
        y_batch = y_batch.reshape(batch_size, seq_length)
        y_predprobility = tf.reshape(y_predprobility, [batch_size, seq_length, -1])
        if len(y) == 0:
            y = y_pred
            y_true = y_batch
            y_prob = y_predprobility
        else:
            y = np.append(y,y_pred,axis=0)
            y_true = np.append(y_true, y_batch, axis=0)
            y_prob = np.append(y_prob, y_predprobility, axis=0)
        
        
        # Check the loss value
        assert not np.isnan(loss_value), \
        "Model diverged with loss = NaN"

    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = y
    total_y_true = y_true
    total_y_prob = y_prob
        
    return total_y_true, total_y_pred, total_loss, duration, total_y_prob

   

def seqcnn(data_dir, model_dir):
    # Make subdirectory for pretraining
    
    optimizer_f = tf.keras.optimizers.Adam(0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    step_f = tf.Variable(1, name='step', trainable=False)
    ckpt_f = tf.train.Checkpoint(step=step_f, optimizer=optimizer_f, net=f_model)
    
    
    optimizer_1 = tf.keras.optimizers.Adam(0)
    optimizer_2 = tf.keras.optimizers.Adam(0)
    loss_s = tf.keras.losses.SparseCategoricalCrossentropy()
    
    f_model.load_weights(model_dir)
    
    
    data_loader_1 = NonSeqDataLoader(data_dir, n_folds, fold_idx, n_classes,seq_length)
    x_train, y_train, x_valid, y_valid = data_loader_1.load_train_data() #这里出来就是（N-19）x20x3840x2的数据
    
    validaccmax = 0
    
    y_true_train, y_pred_train, train_loss, train_duration, y_prob_train = \
                                f_run_epoch(x_train, y_train, batch_size, training=False)
                                
    y_len_train = len(y_true_train)
    # print(y_true_train[0:10,:])
    y_nonseq_train = np.zeros((y_len_train + seq_length - 1))
    for nn in range(seq_length-1):
        y_nonseq_train[nn] = y_true_train[0,nn]
    for nnn in range(y_len_train):
        y_nonseq_train[nnn+seq_length-1] = y_true_train[nnn,seq_length-1]
    
    temp_y_prob_train = np.ones((y_len_train + seq_length - 1, seq_length, n_classes))
    for nnnn in range(seq_length):
        temp_y_prob_train[nnnn:(nnnn+y_len_train),nnnn,:] = y_prob_train[:,nnnn,:]
    # temp_y_prob_train += 1e-8
    temp_y_prob_train = np.log(temp_y_prob_train)
    y_prob_train_sum = np.sum(temp_y_prob_train, axis=1)
    y_nonseq_train_pred = tf.argmax(y_prob_train_sum, axis=1)
    y_nonseq_train_pred.numpy()
    
    y_true_train = y_true_train.reshape(-1,1)
    y_pred_train = y_pred_train.reshape(-1,1)
    
    # print(y_nonseq_train_pred)
    
    
    n_train_examples = len(y_nonseq_train)
    train_cm = confusion_matrix(y_nonseq_train, y_nonseq_train_pred)
    train_acc = np.mean(y_nonseq_train == y_nonseq_train_pred)
    train_f1 = f1_score(y_nonseq_train, y_nonseq_train_pred, average="macro") 
    
    # train_cm = confusion_matrix(y_true_train, y_pred_train)
    # train_acc = np.mean(y_true_train == y_pred_train)
    # train_f1 = f1_score(y_true_train, y_pred_train, average="macro") 
    
    y_true_val, y_pred_val, valid_loss, valid_duration, y_prob_val = \
                            f_run_epoch(x_valid, y_valid, batch_size, training=False)
                            
    y_len_val = len(y_true_val)
    y_nonseq_val = np.zeros((y_len_val + seq_length - 1))
    for nn in range(seq_length-1):
        y_nonseq_val[nn] = y_true_val[0,nn]
    for nnn in range(y_len_val):
        y_nonseq_val[nnn+seq_length-1] = y_true_val[nnn,seq_length-1]  
    
    temp_y_prob_val = np.ones((y_len_val + seq_length - 1, seq_length, n_classes))
    for nnnn in range(seq_length):
        temp_y_prob_val[nnnn:(nnnn+y_len_val),nnnn,:] = y_prob_val[:,nnnn,:]
    temp_y_prob_val = np.log(temp_y_prob_val)
    y_prob_val_sum = np.sum(temp_y_prob_val, axis=1)
    y_nonseq_val_pred = tf.argmax(y_prob_val_sum, axis=1)
    y_nonseq_val_pred.numpy()
    
    y_true_val = y_true_val.reshape(-1,1)
    y_pred_val = y_pred_val.reshape(-1,1)
    
    n_valid_examples = len(y_nonseq_val)
    valid_cm = confusion_matrix(y_nonseq_val, y_nonseq_val_pred)
    valid_acc = np.mean(y_nonseq_val == y_nonseq_val_pred)
    valid_f1 = f1_score(y_nonseq_val, y_nonseq_val_pred, average="macro")
                
    pretrain_epochs = 0
    epoch = 10
    # Report performance
    print_performance(n_train_examples, n_valid_examples, train_cm, 
                      valid_cm, epoch, pretrain_epochs, train_duration, 
                      train_loss, train_acc, train_f1, valid_duration, 
                      valid_loss, valid_acc, valid_f1)    
    
    return y_nonseq_train_pred, y_nonseq_train



