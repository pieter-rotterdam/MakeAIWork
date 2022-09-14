from cmath import exp
import math

#activation softmax function
def softmax():    
   returnX = exp(1) / exp(1) + exp(2)
   returnO = exp(1) / exp(1) + exp(2)

training_inputs = ( ((1,1,1),(1,0,1),(1,1,1)),  #0
    (
        (0,1,0),
        (1,0,1),
        (0,1,0)
    ), #0
    (
        (0,1,0),
        (1,1,1),
        (0,1,0)
    ), #X
    (
        (1,0,1),
        (0,1,0),
        (1,0,1)
    ), #X
)

training_outputs = ([[0,0,1,1]])
training_outputT = 

'''    
m = [[1,2],[3,4],[5,6]]
for row in m :
 print(row)
rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
print("\n")
for row in rez:
 print(row)
'''

print (training_inputs)

inverse_training = [[training_inputs[j][i] for j in range(len (training_inputs))] for i in range (len(training_inputs[0]))]
print("/n")
for row in inverse_training:
    print (row)