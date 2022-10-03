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

d = pd.read_csv ('./simulations/car/control_client/sonar.samples525.csv', sep=' ', names=["dist1","dist2","dist3","angle"])
# d.to_csv('./simulations/car/control_client/sonartest.csv', index=False)
checkpoint_path = "./simulations/car/control_client/angle_prediction2.ckpt"
# print (d)

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

trainLabels = trainDataset.pop('angle')
testLabels = testDataset.pop('angle')
validLabels = validDataset.pop('angle')
lb = LabelEncoder()
trainLabels = lb.fit_transform(trainLabels)
testLabels = lb.fit_transform(testLabels)
validLabels = lb.fit_transform(validLabels)

# test normalizing vs standardizing MinMaxscaler vs StandardScaler, they help the model (less Spiky data)
scaler = StandardScaler()
stdTrainData = scaler.fit_transform(trainDataset)
stdTestData = scaler.fit_transform(testDataset)
stdValidDataset = scaler.fit_transform(validDataset)
stdTrainStats = scaler.fit_transform(trainStats)
# print (stdTrainData)
# print(f"Display the datatype of the test_dataset: {type(stdTrainData)}")
# print(f" Train dataset       : {stdTrainData.shape}")
# breakpoint ()


"""
print (type(stdTrainData.shape[1],))        <class 'int'>
print ((stdTrainData.shape[1]))             3
print (dir((stdTrainData.shape[1])))

   
    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__',
     '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__',
      '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', 
      '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', 
      '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__',
       '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__',
        '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', 
        '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', 
        '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', 
        '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', 
        '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', 
        '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 
        'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']
    """


def build_model2_three_hidden_layers():
 model2 = Sequential() 
 model2.add(Dense(3, input_shape = (stdTrainData.shape[1],)))
 model2.add(Dense(128,Activation('relu'))) 
 model2.add(Dense(256,Activation('relu')))
 model2.add(Dense(512,Activation('relu'))) 
 model2.add(Dense(1))
 learning_rate = 0.001
 optimizer = optimizers.Adam(learning_rate)
 model2.compile(loss='mse', optimizer=optimizer,
 metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
 return model2
 
model2 = build_model2_three_hidden_layers()
#model.load_weights('./model_weights/my_checkpoint') voor testing cascading models
print('Here is a summary of this model: ')
model2.summary()

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
monitor='val_loss', # or val_accuracy if you have it.
save_best_only=True, # Default false. If you don't change the file name then the output will be overritten at each step and only the last model will be saved.
save_weights_only=True, # True => model.save_weights (weights and no structure, you need JSON file for structure), False => model.save (saves weights & structure)
verbose=0,)

EPOCHS = 15 # hoger gaf geen verbetering
batch_size = 32

with tf.device('/CPU:0'): 
  history = model2.fit(
  stdTrainData,
  trainLabels,
  batch_size = batch_size,
  epochs=EPOCHS, 
  verbose=2,
  shuffle=True,
  steps_per_epoch = int(stdTrainData.shape[0] / batch_size),
  validation_data = (stdValidDataset, validLabels),
  callbacks=[tfdocs.modeling.EpochDots(), ckpt_callback],
  )
model2.save('./simulations/car/control_client/sonarmodel2.h5')
model2.save_weights('./model_weights/my_checkpoint2')

example_batch = stdTrainData[:1]
example_result = model2.predict(example_batch)
print('Training predicted values 3 layers: ')
print(example_batch)
print(example_result)

print('The actual labels 3 layers: ')
print (trainLabels[:10])

prediction = model2.predict(stdTrainData)
print (stdTrainData)
print (prediction) # komt eruit als np array

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = 'accuracy')
plt.ylim([0.7, 0.85])
plt.ylabel('Angle [angle]')
plt.show()

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