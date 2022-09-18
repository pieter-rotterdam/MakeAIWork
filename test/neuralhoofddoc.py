# Define datasets (training set should normally be much larger than test set for best results)
          
import math


trainingSet = (
    ((
        (1, 1, 1),
        (1, 0, 1),
        (1, 1, 1)
    ), 'O'),
    ((
        (0, 1, 0),
        (1, 0, 1),
        (0, 1, 0)
    ), 'O'),
    ((
        (0, 1, 0),
        (1, 1, 1),
        (0, 1, 0)
    ), 'X'),
    ((
        (1, 0, 1),
        (0, 1, 0),
        (1, 0, 1)
    ), 'X')
)

testSet = (
    ((
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0)
    ), 'O'),
    ((
        (1, 0, 1),
        (1, 0, 1),
        (1, 1, 0)
    ), 'O'),
    ((
        (1, 0, 0),
        (1, 1, 1),
        (0, 0, 1)
    ), 'X'),
    ((
        (0, 0, 1),
        (1, 1, 1),
        (1, 0, 0)
    ), 'X')
)

# mList = [[0, 0, 1],
#          [1, 1, 1],
#          [1, 0, 0]]

# eList = mList[0][0]

mTup = ((0, 0, 1),
        (1, 1, 1),
        (1, 0, 0))

# eTup = mTup[0][0]

# print(eTup)

def mat2vec(mat):

    # Convert 3 x 3 to 9 x 1
    vec = []

    rows = len(mat)
    
    for row in range(0, rows):

        cols = len(mat[row])

        for col in range(0, cols):

            vec.append(mat[row][col])

    return vec

# print(mTup)

# v = mat2vec(mTup)

# print(v)

# def in2out(input, weights):
#     pass

s =((
        (0, 0, 1),
        (1, 1, 1),
        (1, 0, 0)
    ), 'X')

# print(s[0])
# print(s[1])

def initWeights(vec):

    n = len(vec)

    weights = []

    for i in range(0,n):

        weights.append(1.0)

    return weights


def in2out(vec, weights):

    n = len(vec)

    Sum = 0.0

    # Compute vec[i] * weights[i]
    for i in range(0, n):

        Sum += vec[i] * weights[i]

    # TODO: softmax
    return math.sqrt(Sum)


v = mat2vec(s[0])
w = initWeights(v)
out = in2out(v, w)

print(out)