import time as tm
import traceback as tb # a stack trace prints all the calls prior to the function that raised an exception. In all cases, the last line of a stack trace prints the most valuable information as here the error gets printed.
import math as mt
import sys as ss # The sys module in Python provides various functions and variables that are used to manipulate different parts of the Python runtime environment.
import os # The OS module in Python provides functions for interacting with the operating system b.v. get current directory
import socket as sc # Socket programming is a way of connecting two nodes on a network to communicate with each other. 

ss.path +=  [os.path.abspath (relPath) for relPath in  ('..',)] 

import socket_wrapper as sw
# module die zorgt voor zenden/ontvangen van informatie tussen nodes
import parameters as pm


class SelfcodedClient:
    def __init__ (self):
        self.steeringAngle = 0

        with open (pm.sampleFileName, 'w') as self.sampleFile:
            with sc.socket (*sw.socketType) as self.clientSocket:       #communicatie opzetten
                self.clientSocket.connect (sw.address)
                self.socketWrapper = sw.SocketWrapper (self.clientSocket)
                self.halfApertureAngle = False

                while True:
                    self.input () # via socketwrapper sensor info krijgen
                    self.sweep ()
                    self.output ()
                    self.logTraining ()
                    tm.sleep (0.02)

    def input (self):
        sensors = self.socketWrapper.recv ()

        if not self.halfApertureAngle: # de hoek die de sensor kan zien
            self.halfApertureAngle = sensors ['halfApertureAngle']
            self.sectorAngle = 2 * self.halfApertureAngle / pm.lidarInputDim
            self.halfMiddleApertureAngle = sensors ['halfMiddleApertureAngle']
            
        if 'lidarDistances' in sensors:
            self.lidarDistances = sensors ['lidarDistances']
        else:
            self.sonarDistances = sensors ['sonarDistances']
            
            
            
            
            
            
            def output (self):
        actuators = {
            'steeringAngle': self.steeringAngle,
            'targetVelocity': self.targetVelocity
        }

        self.socketWrapper.send (actuators)