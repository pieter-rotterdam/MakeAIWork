#!/usr/bin/env python

from re import X
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

print(f"Tensorflow version: {tf.version.VERSION}")

d = pd.read_csv('./sonar.csv')
d = d.astype(str)

for itm in d.head():
  d[itm] = d[itm].str.split(',') 
d = d.apply(pd.Series.explode)
# d.to_csv('sonar1.csv', index=False)
tf.random.set_seed(1)
# this is to get same results every expiriment
tf.debugging.set_log_device_placement(False)

d.sample(frac=0.9)
train_dataset = d.sample(frac=0.6)
test_dataset = d.drop(train_dataset.index)
train_dataset, temp_test_dataset =  train_test_split(d, test_size=0.4)
test_dataset, valid_dataset =  train_test_split(temp_test_dataset, test_size=0.5)

train_stats = train_dataset.describe()
train_stats.pop("angle")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('angle')
test_labels = test_dataset.pop('angle')
valid_labels = valid_dataset.pop('angle')

#test normalizing vs standardizing
#normalize
scaler = MinMaxScaler()
normedTrainData = scaler.fit_transform(train_dataset)
normedTestData = scaler.fit_transform(test_dataset)
normedValidDataset = scaler.fit_transform(valid_dataset)
normedTrainStats = scaler.fit_transform(train_stats)
# print (normedTrainStats)
# normalizing helps the algorythms in the model i.e. less spiky  data

#standardize
# scaler = StandardScalar()
# scaler.fit_transform(d)
# standardization gets the mean of values to 0 and the variance to 1

def build_model1_one_hidden_layer():
 model = Sequential() 
 model.add(Dense(8, input_shape = (normedTrainData.shape[1],)))
 model.add(Dense(32,Activation('relu'))) 
 model.add(Dense(1)) 
 
 learning_rate = 0.001
 optimizer = optimizers.Adam(learning_rate)
 model.compile(loss='mse',
 optimizer=optimizer,
 metrics=['mse']) # for regression problems, mean squared error (MSE) is often employed
 return model

model = build_model1_one_hidden_layer()
print('Here is a summary of this model: ')
model.summary()

example_batch = normedTrainData[:10] # take the first 10 data points from the training data.
example_result = model.predict(example_batch)
print (example_result)

# print( train_dataset.shape )
# print( temp_test_dataset.shape )
# print( test_dataset.shape )
# print( valid_dataset.shape )

# print(f"Display the datatype of the test_dataset: {type(test_dataset)}")
# print(f" Train dataset       : {train_dataset.shape}")
# print(f" Test dataset       : {test_dataset.shape}")
# print(f" Validation dataset : {valid_dataset.shape}")
# print(f'No of rows/columns in the dataset: {d.shape}')
# 1529/5

# print (d.info()) 
# print (d.dtypes)
# print(d.head)
# print (train_labels)
# print (type(train_samples))
# print (train_labels)
# print (train_samples) 
# print (d)
