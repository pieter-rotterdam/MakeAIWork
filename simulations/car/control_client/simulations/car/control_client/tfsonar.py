#!/usr/bin/env python

from re import X
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization,  Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

d = pd.read_csv('./sonar.csv', index_col=0)
d = d.astype(str)

for itm in d.head():
  d[itm] = d[itm].str.split(',')
d = d.apply(pd.Series.explode)

# print (d.dtypes)
# d.to_csv('sonar.csv', index=False)

tf.random.set_seed(1)
# this is to get same results every expiriment
# tf.debugging.set_log_device_placement(False)

d.sample(frac=0.9)
trainDataset = d.sample(frac=0.6)
testDataset = d.drop(trainDataset.index)
trainDataset, tempTestDataset =  train_test_split(d, test_size=0.4)
testDataset, validDataset =  train_test_split(tempTestDataset, test_size=0.5)

trainStats = trainDataset.describe()
trainStats.pop('angle')
trainStats = trainStats.transpose()
# print (trainStats)

trainLabels = trainDataset.pop('angle')
testLabels = testDataset.pop('angle')
validLabels = validDataset.pop('angle')
lb = LabelEncoder()
trainLabels = lb.fit_transform(trainLabels)
testLabels = lb.fit_transform(testLabels)
validLabels = lb.fit_transform(validLabels)

# test normalizing vs standardizing
# normalize
scaler = MinMaxScaler()
normedTrainData = scaler.fit_transform(trainDataset)
normedTestData = scaler.fit_transform(testDataset)
normedValidDataset = scaler.fit_transform(validDataset)
normedTrainStats = scaler.fit_transform(trainStats)
# print (normedTrainStats)
# normalizing helps the algorythms in the model i.e. less spiky  data

#standardize
# scaler = StandardScalar()
# scaler.fit_transform(d)
# standardization gets the mean of values to 0 and the variance to 1

# def build_model1_one_hidden_layer():
model = Sequential() 
model.add(Dense(8, input_shape = (normedTrainData.shape[1],)))
model.add(Dense(32,Activation('relu'))) 
model.add(Dense(3)) 
 
# #  learning_rate = 0.001
# #  optimizer = optimizers.Adam(learning_rate)
# #  model.compile(loss='mse', optimizer=optimizer,
# #  metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
# #  return model

model.compile(optimizer='adam',
loss=tf.keras.losses.MeanSquaredError(),
metrics=['accuracy'])
model.fit(normedTrainData, trainLabels, epochs=10)

# accuracy bij metrics gebruiken - hier kun je zien of hij preciezer wordt

# model = build_model1_one_hidden_layer()
# print('Here is a summary of this model: ')
# model.summary()

# example of working code
# example_batch = normedTrainData[:10] # take the first 10 data points from the training data.
# example_result = model.predict(example_batch)
# print (example_result)

# EPOCHS = 1
# batch_size = 32

# with tf.device('/CPU:0'): 
#   history = model.fit(normedTrainData, trainLabels, epochs=10)
   
   
  #  normedTrainData,
  #  trainLabels,
  #  batch_size = batch_size,
  #  epochs=EPOCHS, 
  #  verbose=2,
  #  shuffle=True,
  #  steps_per_epoch = int(normedTrainData.shape[0] / batch_size) ,
  #  validation_data = (normedValidDataset, validLabels),
  #  )

  
  #Ruud stepsPerEpoch divides dataset by batch size
  #stuurhoek waarde tussen -1 en -1

# print(f"Display the datatype of the test_dataset: {type(testDataset)}")
# print(f" Train dataset       : {trainDataset.shape}")
# print(f" Test dataset       : {testDataset.shape}")
# print(f" Validation dataset : {validDataset.shape}")
# print(f'No of rows/columns in the dataset: {d.shape}')
# 1529/5

# print (d.info()) 
# print (d.dtypes)
# print(d.head)
# print (trainLabels)
# print (trainDataset)
# print (type(testDataset))
# print (type(validDataset))
# print (trainLabels)
# print (trainSamples) 
# print (d)
