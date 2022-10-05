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
import keras_tuner as kt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

d = pd.read_csv ('./simulations/car/control_client/sonar.samplesa.csv', sep=' ', names=["dist1","dist2","dist3","angle"])
# d.to_csv('./simulations/car/control_client/sonartest.csv', index=False)
checkpoint_path = "./simulations/car/control_client/angle_prediction1.ckpt"
# print (d)

X=d.iloc[:,:-1] ## data 
y=d.iloc[:,-1] ## labels

trainData,testData,trainLabel,testLabel=train_test_split(X,y, test_size=.15)
trainData,valData,trainLabel,valLabel=train_test_split(trainData,trainLabel, test_size=.17)

tf.random.set_seed(1)
# this is to get same results every expiriment

lb= LabelEncoder()
stdTrainLabel = lb.fit_transform(trainLabel)
stdTestLabel = lb.fit_transform(testLabel)
stdValLabel = lb.fit_transform(valLabel)

# test normalizing vs standardizing MinMaxscaler vs StandardScaler, they help the model (less Spiky data)
scaler = StandardScaler()
stdTrainData = scaler.fit_transform(trainData)
stdTestData = scaler.fit_transform(testData)
stdValData = scaler.fit_transform(valData)
# print (stdTrainData)
# print(f"Display the datatype of the test_dataset: {type(stdTrainData)}")
# print(f" Train dataset       : {stdTrainData.shape}")
# breakpoint ()

def build_tunermodel_five_hidden_layers():
 model = Sequential() 
 model.add(Dense(3, input_shape = (stdTrainData.shape[1],)))
 model.add(Dense(40, activation='relu'))
 model.add(Dense(8, activation='relu'))
 model.add(Dense(40, activation='relu'))
 model.add(Dense(40, activation='relu'))
 model.add(Dense(8, activation='relu'))
 model.add(Dense(1)) 
 
 learning_rate = 0.01
 optimizer = optimizers.Adam(learning_rate)
 model.compile(loss='mse', optimizer=optimizer,
 metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
 return model
# accuracy bij metrics gebruiken - hier kun je zien of hij preciezer wordt

model = build_tunermodel_five_hidden_layers()
print('Here is a summary of this model: ')
model.summary()

example_batch = stdTrainData[:10] # take the first 10 data points from the training data.
example_result = model.predict(example_batch)
example_result

example_batch = stdTrainData[:10]
example_result = model.predict(example_batch)
print('Training predicted values 1 layer: ')
print(example_result)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                                 save_best_only=True, 
                                                 save_weights_only=True, 
                                                 verbose=0,
                                                  )
EPOCHS = 5
batch_size = 32 # 6 iteration

model = build_tunermodel_five_hidden_layers()
print('Here is a summary of this model: ')
model.summary()

with tf.device('/CPU:0'): 
  history = model.fit(
  stdTrainData,
  trainLabel,
  batch_size = batch_size,
  epochs=EPOCHS,
  verbose=2,
  shuffle=True,
  steps_per_epoch = int(stdTrainData.shape[0] / batch_size),
  validation_data = (stdValData, stdValLabel),
  callbacks=[tfdocs.modeling.EpochDots(), ckpt_callback],
  )

example_batch = stdTrainData[:10]
example_result = model.predict(example_batch)
print('predicted values: ')
example_result

print('Traininglabels: ')
trainLabel[:10]

example_batch = stdTestData[:10]
example_result = model.predict(example_batch)
print('predicted values: ')
example_result

print('Testlabels: ')
testLabel[:10]

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = 'accuracy')
plt.ylim([0, 0.85])
plt.ylabel('Angle [angle]')
plt.show()

train_predictions = model.predict(stdTrainData).flatten()

a = plt.axes(aspect='equal')
plt.scatter(trainLabel, test_predictions)
plt.xlabel('True Values angle')
plt.ylabel('Predictions angle')
lims = [0, 60]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(testLabel, test_predictions)
plt.xlabel('True Values angle')
plt.ylabel('Predictions angle')
lims = [0, 60]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

model.save('./simulations/car/control_client/sonarmodel1.h5')
model.save_weights('./model_weights/my_checkpoint1')



# prediction = model.predict(stdTrainData)
# print (stdTrainData)
# print (prediction) # komt eruit als np array



# print(f"Display the datatype of the test_dataset: {type(testDataset)}")
# print(f" Train dataset       : {trainDataset.shape}")
# print(f" Test dataset       : {testDataset.shape}")
# print(f" Validation dataset : {validDataset.shape}")
# print(f'No of rows/columns in the dataset: {d.shape}')


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

#plt.show()