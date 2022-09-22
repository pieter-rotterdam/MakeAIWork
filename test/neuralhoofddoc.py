# Define datasets (training set should normally be much larger than test set for best results)
          
import math

s =((
        (0, 0, 1),
        (1, 1, 1),
        (1, 0, 0)
    ), 'X')

def mat2vec(mat):

    # Convert 3 x 3 to 9 x 1
    vec = []
    rows = len(mat)
    
    for row in range(0, rows):
        cols = len(mat[row])

        for col in range(0, cols):
            vec.append(mat[row][col])

    return vec

# print(s[0])
# print(s[1])

def initWeights(vec):

    n = len(vec)
    weights = []

    for i in range(0,n):
        weights.append(1.0)

    return weights

def in2out
:

    n = len(vec)
    Sum = 0.0

    # Compute vec[i] * weights[i]
    for i in range(n):
        Sum += vec[i] * weights[i]
    return (Sum)

print (in2out)

v = mat2vec(s[0])
w = initWeights(v)
out = in2out(v, w)

print (out)