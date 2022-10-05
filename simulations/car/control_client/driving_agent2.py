'''
====== Legal notices

Copyright (C) 2013 - 2021 GEATEC engineering

This program is free software.
You can use, redistribute and/or modify it, but only under the terms stated in the QQuickLicense.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY, without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the QQuickLicense for details.

The QQuickLicense can be accessed at: http://www.qquick.org/license.html

__________________________________________________________________________


 THIS PROGRAM IS FUNDAMENTALLY UNSUITABLE FOR CONTROLLING REAL SYSTEMS !!

__________________________________________________________________________

It is meant for training purposes only.

Removing this header ends your license.
'''

import time as tm
import traceback as tb # a stack trace prints all the calls prior to the function that raised an exception. In all cases, the last line of a stack trace prints the most valuable information as here the error gets printed.
import math as mt
import sys as ss # The sys module in Python provides various functions and variables that are used to manipulate different parts of the Python runtime environment.
import os # The OS module in Python provides functions for interacting with the operating system b.v. get current directory
import socket as sc # Socket programming is a way of connecting two nodes on a network to communicate with each other. 
import tensorflow as tf


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

ss.path +=  [os.path.abspath (relPath) for relPath in  ('..',)] 

import socket_wrapper as sw
# module die zorgt voor zenden/ontvangen van informatie tussen nodes
import parameters as pm

model_sonar_path = 'simulations/car/control_client/sonarmodel1.h5'

class HardcodedClient:
    def __init__ (self):
        self.model = None
        self.steeringAngle = 0
        self.targetVelocity = 0

        with open (pm.sampleFileName, 'w') as self.sampleFile:
            with sc.socket (*sw.socketType) as self.clientSocket:       #communicatie opzetten
                self.clientSocket.connect (sw.address)
                self.socketWrapper = sw.SocketWrapper (self.clientSocket)
                self.halfApertureAngle = False

                while True:
                    self.input () 
                    self.sweep ()
                    self.output ()
                    self.logTraining ()
                    tm.sleep (0.02)

    def input (self):
        sensors = self.socketWrapper.recv () # via socketwrapper sensor info krijgen

        if not self.halfApertureAngle: # de hoek die de sensor kan zien
            self.halfApertureAngle = sensors ['halfApertureAngle']
            self.sectorAngle = 2 * self.halfApertureAngle / pm.lidarInputDim
            self.halfMiddleApertureAngle = sensors ['halfMiddleApertureAngle']
            
        if 'lidarDistances' in sensors:
            self.lidarDistances = sensors ['lidarDistances']
            if self.model==None:
                self.model = tf.keras.models.load_model(model_lidar_path)           
        else:
            self.sonarDistances = sensors ['sonarDistances']

            if self.model==None:
                self.model = tf.keras.models.load_model(model_sonar_path) 

    def lidarSweep (self):
        nearestObstacleDistance = pm.finity
        nearestObstacleAngle = 0
        
        nextObstacleDistance = pm.finity
        nextObstacleAngle = 0

        for lidarAngle in range (-self.halfApertureAngle, self.halfApertureAngle):
            lidarDistance = self.lidarDistances [lidarAngle]
            
            if lidarDistance < nearestObstacleDistance:
                nextObstacleDistance =  nearestObstacleDistance
                nextObstacleAngle = nearestObstacleAngle
                
                nearestObstacleDistance = lidarDistance 
                nearestObstacleAngle = lidarAngle

            elif lidarDistance < nextObstacleDistance:
                nextObstacleDistance = lidarDistance
                nextObstacleAngle = lidarAngle
           
        targetObstacleDistance = (nearestObstacleDistance + nextObstacleDistance) / 2

        self.steeringAngle = (nearestObstacleAngle + nextObstacleAngle) / 2
        self.targetVelocity = pm.getTargetVelocity (self.steeringAngle)
     

    def sonarSweep (self):

        steering_angle_model = self.model.predict([self.sonarDistances])
        self.steeringAngle = float(steering_angle_model[0][0])
        if self.steeringAngle < 0.5: self.steeringAngle = -22
        elif self.steeringAngle < 1.5: self.steeringAngle = 0
        elif self.steeringAngle > 1.5: self.steeringAngle = 22
        self.targetVelocity = pm.getTargetVelocity (self.steeringAngle)
        
    def sweep (self):
        if hasattr (self, 'lidarDistances'):
            self.lidarSweep ()
        else:
            self.sonarSweep ()

    def output (self):
        actuators = {
            'steeringAngle': self.steeringAngle,
            'targetVelocity': self.targetVelocity
        }

        self.socketWrapper.send (actuators)

    def logLidarTraining (self):
        sample = [pm.finity for entryIndex in range (pm.lidarInputDim + 1)]

        for lidarAngle in range (-self.halfApertureAngle, self.halfApertureAngle):
            sectorIndex = round (lidarAngle / self.sectorAngle)
            sample [sectorIndex] = min (sample [sectorIndex], self.lidarDistances [lidarAngle])

        sample [-1] = self.steeringAngle
        print (*sample, file = self.sampleFile)

    def logSonarTraining (self):
        sample = [pm.finity for entryIndex in range (pm.sonarInputDim + 1)]

        for entryIndex, sectorIndex in ((2, -1), (0, 0), (1, 1)):
            sample [entryIndex] = self.sonarDistances [sectorIndex]

        sample [-1] = self.steeringAngle
        print (*sample, file = self.sampleFile)

    def logTraining (self):
        if hasattr (self, 'lidarDistances'):
            self.logLidarTraining ()
        else:
            self.logSonarTraining ()

HardcodedClient ()
