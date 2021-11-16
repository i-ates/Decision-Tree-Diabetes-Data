import numpy as np
import datetime  # The library we use to calculate the time difference between algorithms

# This functios does 5foldcrossvalidation for ID3 decision tree classification. It takes dataFrame as a numpy array.
# It split the dataFrame to test Data(%80) and train Data(%20). This function splits
# the dataFrame in to 5 part and takes them respectively each part to test data and take rest of them to train data.
# It calculates the prediction and get accuracy for each part and and avarage accuracy for decision tree.
# After each part calculations it prints the results.
import decisionTree


def kFoldCrossValidationClasification(df,pre = False,post = False):
    # splitNum = rounded result of to division of sample count to 5. It is  size of each 5 part
    splitNum = df.shape[0] // 5
    avgAccKnn = 0  # avarage result of knn classification
    avgCompileTimeKnn = 0  # avarage compile time for knn classification

    # for each splitted part of dataFrame
    for i in range(0, 5):
        trainData = np.copy(df)  # copy of DataFrame
        testData = np.empty(shape=(splitNum, df.shape[1]), dtype=int)  # creating test Data array
        # copying samples to test Data and add their indexes to indexList
        for y in range(splitNum * i, splitNum * (i + 1)):
            if i != 0:
                testData[y - (splitNum * i)] = df[y]
            else:
                testData[y] = df[y]
        # for  j in range to reversed for loop, delete samples in train Data which are added to test Data
        for j in range(splitNum - 1, -1, -1):
            trainData = np.delete(trainData, (j + (splitNum * i)), 0)

        truePredictCount = 0

        dt = decisionTree.Decision_Tree(1e-16, ["x[" + str(i) + "]" for i in range(16)])
        X = trainData[:, :-1]
        y = trainData[:, -1]
        dt.fit(X, y, pre, post)
        start_time = datetime.datetime.now()

        # for each sample in test Data
        testX = testData[:, :-1]
        testy = testData[:, -1]
        acc = dt.accuracy(testX,testy)

        end_time = datetime.datetime.now()
        avgCompileTimeKnn = avgCompileTimeKnn + ((end_time - start_time).total_seconds() * 1000)
        print("Fold " + str(i) + " accurancy:" + str(
            acc))
        # add accuracy to average accuracy
        avgAccKnn = avgAccKnn + acc

    print("")
    print("")
    print("////////////////////////")
    print("Avarage Acc = " + str(avgAccKnn / 5))
    print("Average Run Time = " + str(avgCompileTimeKnn / 5))
    print("////////////////////////")
    print("")
    print("")
