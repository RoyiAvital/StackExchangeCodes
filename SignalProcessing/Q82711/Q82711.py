
# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# SK Learn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

# Keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model

# Miscellaneous
import random
import warnings
from sys import modules
from time import time
import datetime
import os
from platform import python_version
import scipy.io as sio

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


#%% Configuration 

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

#<! Check if GPU is available
#<! See also https://www.tensorflow.org/guide/gpu

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))



# %% Load Train Data

dMatFile = sio.loadmat('TrainSet.mat')
mData = dMatFile['mData']
vLabels = np.squeeze(dMatFile['vLabels'], axis = 1) #<! Labels {0, 1, ..., 99}
vSignal = np.squeeze(dMatFile['vSignal'], axis = 1)

mX = np.transpose(mData, axes = [1, 0]) #<! Batches before channels
mX = np.float32(mX[..., np.newaxis])
vY = np.float32(vLabels - 1) #<! Labels {0, 1, ..., 99}


#%% Load Validation Data

dMatFile = sio.loadmat('ValidationSet.mat')
mData = dMatFile['mData']
vLabels = np.squeeze(dMatFile['vLabels'], axis = 1) #<! Labels {0, 1, ..., 99}
vSignal = np.squeeze(dMatFile['vSignal'], axis = 1)

mXVal = np.transpose(mData, axes = [1, 0]) #<! Batches before channels
mXVal = np.float32(mXVal[..., np.newaxis])
vYVal = np.float32(vLabels - 1) #<! Labels {0, 1, ..., 99}


# %% Keras Model

modelNet = keras.Sequential()
# modelNet.add(keras.Input(shape = (mData.shape[0], 1)))
modelNet.add(keras.layers.Conv1D(32, vSignal.shape[0], activation = 'relu'))
modelNet.add(keras.layers.Conv1D(48, 51, activation = 'relu'))
modelNet.add(keras.layers.Conv1D(64, 25, activation = 'relu'))
modelNet.add(keras.layers.Conv1D(64, 13, activation = 'relu'))
modelNet.add(keras.layers.Flatten())
modelNet.add(keras.layers.Dense(units = 100, activation = 'softmax'))

# modelNet.summary()


#%% Defining CheckPoints

folderName = 'CheckPoints'
checkPointFileName = r'{epoch:03d}-{val_accuracy:.4f}.hdf5'
os.mkdir(folderName) #<! Make sure the folder exist in case of a file
checkPointsFilePath = os.path.join(folderName, checkPointFileName)
modelCheckPointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkPointsFilePath, #<! File to save
    save_weights_only = True, #<! Don't save the whole model but weights only
    monitor = 'val_accuracy', #<! Measure of performance
    mode = 'max', #<! How to compare performance as in `monitor`
    save_best_only = True) #<! Save it only if it is better than current


# %% Model Compilation

initLr      = 0.00045 #<! Relatively high
lrSchedule  = keras.optimizers.schedules.ExponentialDecay(initLr, decay_steps = 1407, decay_rate = 0.96, staircase = True) #<! Today we have much more effective policies
hOpt        = keras.optimizers.Adam(learning_rate = lrSchedule, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False, name = "Adam")

modelNet.compile(optimizer = hOpt, loss = keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy']) #<! A scheduler automatically create a TensorBoard callback

logDir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# In order to run it from command line: tensorboard --logdir logs
callBackTB = tf.keras.callbacks.TensorBoard(logDir, histogram_freq = 1) #<! TensorBoard Callback

# %% Training

batchSize = 48
numEpochs = 21

# modelNet.fit(X_train, y_train, batch_size = batchSize, epochs = numEpochs, verbose = 1, validation_data = (X_test, y_test), callbacks = [callBackTB])
modelNet.fit(mX, vY, batch_size = batchSize, epochs = numEpochs, verbose = 1, validation_data = (mXVal, vYVal), callbacks = [callBackTB, modelCheckPointCallback])

# %% Predicting

# modelNet.load_weights(os.path.join('CheckPoints', '009-0.3013.hdf5'))

# mY = modelNet.predict(mXVal)
# vYPred = np.argmax(mY, axis = 1)
# dMatFileSave = {'vYPred': vYPred}
# sio.savemat('PythonData.mat', dMatFileSave) #<! Remember to add 1 in MATLAB