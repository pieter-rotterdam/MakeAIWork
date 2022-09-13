# youtube.com/watch?v=kft1AJ9WVDk
# https://www.youtube.com/watch?v=Py4xvZx-A1E

#snippet
"""
def create_neural_net(layer_array, input_dims):
    weights = []
    biases = []
    activations = []
    
    for i in range(len(layer_array)):
        node_num = layer_array[i][0]
        weights_of_layer = []
        biases_of_layer = []
        if i == 0:
            last_layer_node_number = input_dims
        else:
            last_layer_node_number = layer_array[i-1][0]
        
        for n in range(0,node_num):
            weights_of_node = []
            for l in range(0, last_layer_node_number):
                weights_of_node.append(1) 
            weights_of_layer.append(weights_of_node)
            biases_of_layer.append(0)
            
        weights.append(weights_of_layer)
        biases.append(biases_of_layer)
        activations.append(layer_array[i][1])
    return [weights, biases, activations]
"""

#neural network without hidden layers is perceptron

import math

symbolVecs = {'0': (1, 0)}, 'X': (0, 1)} # een nul is T or F en een X is T or F
symbolChars = dict ((value, key) for key, value in symbolVecs.items ())
    1
trainingSet = (
    ((
        (1,1,1),
        (1,0,1),
        (1,1,1)
    ), '0'),
    ((
      ((
        (0,1,0),
        (1,0,1),
        (0,1,0)
    ), '0'),
    ((
    ((
        (0,1,0),
        (1,1,1),
        (0,1,0)
    ), 'X'),
    ((
      ((
        (1,0,1),
        (0,1,0),
        (1,0,1)
    ), 'X')
)
     
trainingSetFalse = (
    ((
        (1,1,1),
        (1,0,1),
        (1,1,1)
    ), '0'),
    ((
      ((
        (0,1,0),
        (1,0,1),
        (0,1,0)
    ), '0'),
    ((
    ((
        (0,1,0),
        (1,1,1),
        (0,1,0)
    ), 'X'),
    ((
      ((
        (1,0,1),
        (0,1,0),
        (1,0,1)
    ), 'X')
)   

camera_input = (
    ((
        (0,0,0),
        (0,0,0),
        (0,0,0)
    ), 'camera_input'),
    
output = q

class Node:
    def __init__(self)
    self.value = None
    self.inLinks=[]

class Link:
    def __init__(self,inputNodes,outputNodes,self.weight)
    self.weight = 1 #testwaarde 1 is ok

#activation function
def softmax():
	e = exp(vector)
	return e / e.sum()

#iteration
for iteration in range(1):
    input_layer = training_set
    outputs = softmax



 
# error is 0,5 dus squared is 0.25 
# dan squared error - hoe fout doe je het bij 1
# doe 4x, dan delen door 4 in dit voorbeeld is 4x.5/4=0.5x§

