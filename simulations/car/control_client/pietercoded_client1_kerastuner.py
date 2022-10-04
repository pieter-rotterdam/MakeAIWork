#!/usr/bin/env python

from re import X
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.layers import Activation, Dense, BatchNormalization,  Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

hp = kt.HyperParameters()
tf.random.set_seed(0)

d = pd.read_csv ('./simulations/car/control_client/sonar.samples.csv', sep=' ', names=["dist1","dist2","dist3","angle"])
# d.to_csv('./simulations/car/control_client/sonartest.csv', index=False)

#print(d.head())

d.sample(frac=0.9) #shuffle
trainDataset = d.sample(frac=0.6)
trainDataset, tempTestDataset =  train_test_split(d, test_size=0.4)
testDataset, validDataset =  train_test_split(tempTestDataset, test_size=0.5)

trainLabels = trainDataset.pop('angle')
testLabels = testDataset.pop('angle')
validLabels = validDataset.pop('angle')
lb = LabelEncoder()
stdTrainLabels = lb.fit_transform(trainLabels)
stdTestLabels = lb.fit_transform(testLabels)
stdValidLabels = lb.fit_transform(validLabels)

# test normalizing vs standardizing MinMaxscaler vs StandardScaler, they help the model (less Spiky data)
scaler = StandardScaler()
stdTrainData = scaler.fit_transform(trainDataset)
stdTestData = scaler.fit_transform(testDataset)
stdValidData = scaler.fit_transform(validDataset)

from tensorflow import keras
from tensorflow.keras import layers


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="linear"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    model.compile(
        optimizer="adam", loss="mse", metrics=["accuracy"],
    )
    return model
    
build_model(kt.HyperParameters())

# You may choose from RandomSearch, BayesianOptimization and Hyperband,
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory=('./simulations/car/control_client/'),
    project_name="tuner",
)

tuner.search_space_summary()

tuner.search(stdTrainData, stdTrainLabels, epochs=200, validation_data=(stdValidData, stdValidLabels))

