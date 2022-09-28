#!/usr/bin/env python

from re import X
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization,  Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# print(f"Tensorflow version: {tf.version.VERSION}")

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
normedTrainStats = scaler.fit_transform(train_stats)
print (normedTrainStats)
# normalizing helps the algorythms in the model i.e. less spiky  data

#standardize
# scaler = StandardScalar()
# scaler.fit_transform(d)
# standardization gets the mean of values to 0 and the variance to 1




# print( train_dataset.shape )
# print( temp_test_dataset.shape )
# print( test_dataset.shape )
# print( valid_dataset.shape )

# print(f"Display the datatype of the test_dataset: {type(test_dataset)}")
# print(f" Trai dataset       : {train_dataset.shape}")
# print(f" Test dataset       : {test_dataset.shape}")
# print(f" Validation dataset : {valid_dataset.shape}")



# print(f'No of rows/columns in the dataset: {d.shape}')
# 1529/5

train_samples=d.loc[:,'dist1':'dist3'] 

# print (d.info()) 
# print (d.dtypes)
# print(d.head)
# print (train_labels)
# print (type(train_samples))
# print (train_labels)
# print (train_samples) 
# print (d)

# Model = tf.keras.Sequential([
#   Dense(units=3, activation='relu'),
#   Dense(units=1, activation='relu')])

# voor ons model input = 3 of 16, de input wordt dus gegeven aan eerste hidden layer

# Model.summary()

# tf.convert_to_tensor(train_samples)
''''
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers


filename = "C:/MakeAIWork/simulations/car/control_client/sonar.csv"

sonar_01_train = pd.read_csv(filename, header = 0, sep = ',')
# print(sonar_01_train)
# type(sonar_01_train)

sonar_01_train.head()

sonar_01_sensors = sonar_01_train.copy()
sonar_01__labels = sonar_01_sensors.pop('Steering Angle')

# print(sonar_01__labels)

sonar_01_sensors = np.array(sonar_01_sensors)

# print(sonar_01_sensors)

sonar_01_model = tf.keras.Sequential([
  layers.Dense(8),
  layers.Dense(1)
])

sonar_01_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(0.1))

sonar_01_model.fit(sonar_01_sensors, sonar_01__labels, epochs=3)
'''