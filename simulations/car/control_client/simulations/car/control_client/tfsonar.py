#!/usr/bin/env python

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# SHUFFLE_BUFFER = 500
# BATCH_SIZE = 2

d = pd.read_csv('sonar.csv')
d = d.astype(str)

for itm in d.head():
  d[itm] = d[itm].str.split(',')
  
d = d.apply(pd.Series.explode)

print(d.head)

d.to_csv('sonar1.csv', index=False)

train_labels=d.loc[:,'angle']
train_samples=d.loc[:,'dist1':'dist3'] 

print (train_labels)
# print (type(train_samples))
# print (train_labels)
# print (train_samples) 
# print (d)

# tf.convert_to_tensor(train_samples)

# Model = tf.keras.Sequential([
#   Dense(units=3, activation='relu'),
#   Dense(units=1, activation='relu')])



# voor ons model input = 3 of 16, de input wordt dus gegeven aan eerste hidden layer

# model.summary()

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