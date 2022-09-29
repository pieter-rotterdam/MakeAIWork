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
from tensorflow.keras.metrics import categorical_accuracy
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

def build_model1_one_hidden_layer():
 model = Sequential() 
 model.add(Dense(3, input_shape = (normedTrainData.shape[1],)))
 model.add(Dense(32,Activation('relu'))) 
 model.add(Dense(1)) 
 
 learning_rate = 0.001
 optimizer = optimizers.Adam(learning_rate)
 model.compile(loss='mse', optimizer=optimizer,
 metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
 return model
# accuracy bij metrics gebruiken - hier kun je zien of hij preciezer wordt
model = build_model1_one_hidden_layer()
print('Here is a summary of this model: ')
model.summary()


def build_model2_three_hidden_layers():
 model = Sequential() 
 model.add(Dense(3, input_shape = (normedTrainData.shape[1],)))
 model.add(Dense(32,Activation('relu'))) 
 model.add(Dense(64,Activation('relu')))
 model.add(Dense(128,Activation('relu'))) 
 model.add(Dense(1)) 
 
 learning_rate = 0.001
 optimizer = optimizers.Adam(learning_rate)
 model.compile(loss='mse', optimizer=optimizer,
 metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
 return model

model2 = build_model2_three_hidden_layers()
print('Here is a summary of this model: ')
model2.summary()

def build_model3_five_hidden_layers():
 model = Sequential() 
 model.add(Dense(3, input_shape = (normedTrainData.shape[1],)))
 model.add(Dense(64,Activation('relu'))) 
 model.add(Dense(64,Activation('relu')))
 model.add(Dense(64,Activation('relu'))) 
 model.add(Dense(64,Activation('relu'))) 
 model.add(Dense(64,Activation('relu'))) 
 model.add(Dense(1)) 
 
 learning_rate = 0.001
 optimizer = optimizers.Adam(learning_rate)
 model.compile(loss='mse', optimizer=optimizer,
 metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
 return model

model3 = build_model3_five_hidden_layers()
print('Here is a summary of this model: ')
model3.summary()

# example of working code
# example_batch = normedTrainData[:10] # take the first 10 data points from the training data.
# example_result = model.predict(example_batch)
# print (example_result)

EPOCHS = 300
batch_size = 32

with tf.device('/CPU:0'): 
  history = model.fit(
  normedTrainData,
  trainLabels,
  batch_size = batch_size,
  epochs=EPOCHS, 
  verbose=2,
  shuffle=True,
  steps_per_epoch = int(normedTrainData.shape[0] / batch_size) ,
  validation_data = (normedValidDataset, validLabels),
  )
  
example_batch = normedTrainData[:10]
example_result = model.predict(example_batch)
print('Training predicted values one layer: ')
print(example_result)

print('The actual labels one layer: ')
print (trainLabels[:10])

example_batch = normedTestData[:10]
example_result = model.predict(example_batch)
print('Testing predicted values one layer: ')
print(example_result)

print('The actual labels one layer: ')
print (testLabels[:10])

EPOCHS = 300
batch_size = 32

with tf.device('/CPU:0'): 
  history = model2.fit(
  normedTrainData,
  trainLabels,
  batch_size = batch_size,
  epochs=EPOCHS, 
  verbose=2,
  shuffle=True,
  steps_per_epoch = int(normedTrainData.shape[0] / batch_size) ,
  validation_data = (normedValidDataset, validLabels),
  )
  
example_batch = normedTrainData[:10]
example_result = model.predict(example_batch)
print('Training predicted values three layers: ')
print(example_result)

print('The actual labels three layers: ')
print (trainLabels[:10])

example_batch = normedTestData[:10]
example_result = model.predict(example_batch)
print('Testing predicted values five layers: ')
print(example_result)

print('The actual labels five layers: ')
print (testLabels[:10])

with tf.device('/CPU:0'): 
  history = model3.fit(
  normedTrainData,
  trainLabels,
  batch_size = batch_size,
  epochs=EPOCHS, 
  verbose=2,
  shuffle=True,
  steps_per_epoch = int(normedTrainData.shape[0] / batch_size) ,
  validation_data = (normedValidDataset, validLabels),
  )
  
example_batch = normedTrainData[:10]
example_result = model.predict(example_batch)
print('Training predicted values five layers: ')
print(example_result)

print('The actual labels five layers: ')
print (trainLabels[:10])

example_batch = normedTestData[:10]
example_result = model.predict(example_batch)
print('Testing predicted values five layers: ')
print(example_result)

print('The actual labels five layers: ')
print (testLabels[:10])


  #Ruud #stuurhoek waarde tussen -1 en -1 is je gewenste uitkomst

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
