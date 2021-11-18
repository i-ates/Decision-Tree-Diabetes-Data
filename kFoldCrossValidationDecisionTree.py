import numpy as np
import datetime  # The library we use to calculate the time difference between algorithms

# This functios does 5foldcrossvalidation for ID3 decision tree classification. It takes dataFrame as a numpy array.
# It split the dataFrame to test Data(%80) and train Data(%20). This function splits
# the dataFrame in to 5 part and takes them respectively each part to test data and take rest of them to train data.
# It calculates the prediction and get accuracy for each part and and avarage accuracy for decision tree.
# After each part calculations it prints the results.
import decisionTree


def kFoldCrossValidationClasification(df, pre=False):
    # splitNum = rounded result of to division of sample count to 5. It is  size of each 5 part
    splitNum = df.shape[0] // 5
    avgAcc = 0  # avarage result of knn classification
    avgPrecision = 0
    avgF1Score = 0
    avgRecall = 0
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

        dt = decisionTree.DecisionTree(1e-16, ["x[" + str(i) + "]" for i in range(16)])
        X = trainData[:, :-1]
        y = trainData[:, -1]
        start_time = datetime.datetime.now()
        dt.fit(X, y, pre)
        # for each sample in test Data
        testX = testData[:, :-1]
        testy = testData[:, -1]
        acc = dt.accuracy(testX, testy)
        TP = acc[1]
        FP = acc[3]
        FN = acc[4]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = (2 * (recall * precision)) / (recall + precision)
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
        avgF1Score = avgF1Score + f1score
        end_time = datetime.datetime.now()
        avgCompileTimeKnn = avgCompileTimeKnn + ((end_time - start_time).total_seconds() * 1000)

        print()
        # add accuracy to average accuracy
        avgAcc = avgAcc + acc[0]

    print("")
    print("")
    print("////////////////////////")
    print("Avarage Acc = " + str(avgAcc / 5))
    print("Avarage Precision = " + str(avgPrecision/5))
    print("Avarage Recall = " + str(avgRecall/5))
    print("Avarage F1Score = " + str(avgF1Score/5))
    print("Average Run Time = " + str(avgCompileTimeKnn / 5))
    print("////////////////////////")
    print("")
    print("")
