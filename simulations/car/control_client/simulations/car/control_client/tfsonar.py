import numpy as np
import pandas as pd

d = pd.read_csv('sonar.csv')
d = d.astype(str)

for itm in d.head():
  d[itm] = d[itm].str.split(',')
  
d = d.apply(pd.Series.explode)
# print(d.head)

# d.to_csv('sonar1.csv', index=False)

train_labels=d.loc[:,'angle']
train_samples=d.loc[:,'dist1','dist2','dist3'] # samples nog debuggen om correct weer te geven

#print (train_labels)
print (train_samples) 


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