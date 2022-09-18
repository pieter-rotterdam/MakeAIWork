import math
from telnetlib import VT3270REGIME

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

    Sum1 = 0.0
    Sum2 = 0.0
    Sum3 = 0.0
    Sum4 = 0.0

    # Compute vec[i] * weights[i]
    for i in range(0, n):

        Sum1 += train1[i] * weights[i]
        Sum2 += train2[i] * weights[i]
        Sum3 += train3[i] * weights[i]
        Sum4 += train4[i] * weights[i]

def softmax(inputs):
    temp = [math.exp(v) for v in inputs]
    total = sum(temp)
    return [t / total for t in temp]

act1 = softmax(train1)
for a1 in act1:
    print(f"{a1:.8f}")

print(f"total: {sum(act1)}")

act2 = softmax(train2)
for a2 in act2:
    print(f"{a2:.8f}")

print(f"total: {sum(act2)}")

act3 = softmax(train3)
for a3 in act3:
    print(f"{a3:.8f}")

print(f"total: {sum(act3)}")

act4 = softmax(train4)
for a4 in act4:
    print(f"{a4:.8f}")

print(f"total: {sum(act4)}")
    


v1 = mat2vec(trainingSet[0][0])
v2 = mat2vec(trainingSet[1][0])
v3 = mat2vec(trainingSet[2][0])
v4 = mat2vec(trainingSet[3][0])
w1 = initWeights(v1)
w2 = initWeights(v2)
w3 = initWeights(v3)
w4 = initWeights(v4)
out1 = in2out(v1, w1)
out2 = in2out(v2, w2)
out3 = in2out(v3, w3)
out4 = in2out(v4, w4)

print(out1 , out2, out3, out4)