
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
train1= mat2vec(trainingSet[0][0])
train2= mat2vec(trainingSet[1][0])
train3= mat2vec(trainingSet[2][0])
train4= mat2vec(trainingSet[3][0])
test1= mat2vec(testSet[0][0])
test2= mat2vec(testSet[1][0])
test3= mat2vec(testSet[2][0])
test4= mat2vec(testSet[3][0])

'''




for x in trainingSet: #aanroepen specifieke matrices
    
    train1=(x[0][0])
    train2=(x[0][1])
    train3=(x[0][2])
    train4=(x[0][3])
   
    print (train1)

#train1= 
'''
