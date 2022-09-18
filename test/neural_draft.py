from cmath import exp
import math
'''
#activation softmax function
def softmax():    
   returnX = exp(1) / exp(1) + exp(2)
   returnO = exp(1) / exp(1) + exp(2)'''
   
node_inputs = ( ((1,1,1),(1,0,1),(1,1,1)), #0
                    ((0,1,0),(1,0,1),(0,1,0)), #0
                    ((0,1,0),(1,1,1),(0,1,0)), #X
                    ((1,0,1),(0,1,0),(1,0,1)), #X je geeft de trainingsset, en de testset zodat je kunt zien hoe goed hij leert
)
"""
def note_inputs_flat:
    [j for sub in [node_inputs] for j in sub]
    return
"""


# eerst 18 inputs x gewichten, optellen, daar softmax overheen

node_outputs = ([[0,0,1,1]])

def node_inputs:
    rows = len(training_inputs)
    columns = len(training_inputs[0])

    node_outputs_v = []
    for j in range(columns):
        row = []
        for i in range(rows):
           row.append(node_inputs[i][j])
        node_inputs_v.append(row)

    return node_outputs_v

input= "node inputs"
print (input, node_inputs)
output= "node outputs"
print (output, node_outputs_v)

# Link = 2*np.random.random((3,1)) - 1

# 'gewogen som' z = matrix nodes x matrix gewichten


w = w+yx




'''
rows, cols = 4,1
my_matrix = [([0]*cols) for i in range(rows)]
print (my_matrix)


def fit(self, X, y, n_iter=100):

    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Add 1 for the bias term
    self.weights = np.zeros((n_features+1,))

    # Add column of 1s
    X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

    for i in range(n_iter):
        for j in range(n_samples):
            if y[j]*np.dot(self.weights, X[j, :]) <= 0:
                self.weights += y[j]*X[j, :]
