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

def mat2vec(mat):

    # Convert 3 x 3 to 9 x 1
    vec = []

    rows = len(mat)
    
    for row in range(0, rows):

        cols = len(mat[row])

        for col in range(0, cols):

            vec.append(mat[row][col])

    return vec



matTrainingSet= (trainingSet[0][0],trainingSet[1][0],trainingSet[2][0],trainingSet[3][0])

#print (matTrainingSet)

train1= mat2vec(trainingSet[0][0])
train2= mat2vec(trainingSet[1][0])
train3= mat2vec(trainingSet[2][0])
train4= mat2vec(trainingSet[3][0])
test1= mat2vec(testSet[0][0])
test2= mat2vec(testSet[1][0])
test3= mat2vec(testSet[2][0])
test4= mat2vec(testSet[3][0])

trainList= (train1,train2,train3,train4)
testList= (test1,test2,test3,test4)

#print (trainList)
#print (testList)   

#print (len(train1))


def initWeights(train1):

    n = len(train1)

    weights = []

    for i in range(0,n):

        weights.append(1.0)

    return weights


def in2out(train1, weights):

    n = len(train1)

    Sum = 0.0

    # Compute vec[i] * weights[i]
    for i in range(0, n):

        Sum += train1[i] * weights[i]

    # TODO: softmax
    return math.sqrt(Sum)


v = mat2vec(trainingSet[0][0])
w = initWeights(v)
out = in2out(v, w)

print(out)