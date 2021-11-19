import numpy as np
import datetime  # The library we use to calculate the time difference between algorithms

# This functios does 5foldcrossvalidation for ID3 decision tree classification. It takes dataFrame as a numpy array.
# It split the dataFrame to test Data(%80) and train Data(%20). This function splits
# the dataFrame in to 5 part and takes them respectively each part to test data and take rest of them to train data.
# It calculates the prediction and get accuracy for each part and and avarage accuracy for decision tree.
# After each part calculations it prints the results.


import id3
import node


def kFoldCrossValidationClasification(df):
    # splitNum = rounded result of to division of sample count to 5. It is  size of each 5 part
    splitNum = len(df) // 5
    avgAcc = 0
    avgAccPrun = 0
    avgPrecision = 0
    avgF1Score = 0
    avgRecall = 0
    avgPrecisionPrun = 0
    avgRecallPrun = 0
    avgF1ScorePrun = 0
    avgCompileTime = 0
    avgCompileTimePrun = 0
    for i in range(0, 5):
        trainDict = df.copy()
        testData = []
        # copying samples to test Data and add their indexes to indexList
        for y in range(splitNum * i, splitNum * (i + 1)):
            testData.append(df[y])
        # for  j in range to reversed for loop, delete samples in train Data which are added to test Data
        for j in range(splitNum - 1, -1, -1):
            del trainDict[j + (splitNum * i)]


        validationSize = len(trainDict)//4
        trainData = trainDict[0:len(trainDict)-validationSize]
        validationData = trainDict[len(trainDict)-validationSize:]

        startTime = datetime.datetime.now()
        tree = id3.ID3(trainData ,0)
        acc = id3.test(tree, testData)
        endTime = datetime.datetime.now()
        avgCompileTime = avgCompileTime + ((endTime - startTime).total_seconds() * 1000)

        TP = acc[1]
        FP = acc[3]
        FN = acc[4]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = (2 * (recall * precision)) / (recall + precision)
        avgPrecision = avgPrecision + precision
        avgRecall = avgRecall + recall
        avgF1Score = avgF1Score + f1score
        avgAcc = acc[0] + avgAcc
        print(acc[0])

        startTime = datetime.datetime.now()

        treePruned = id3.prune(tree, validationData)
        accPrun= id3.test(treePruned,testData)
        endTime = datetime.datetime.now()
        avgCompileTimePrun = avgCompileTimePrun + ((endTime - startTime).total_seconds() * 1000)
        TP = accPrun[1]
        FP = accPrun[3]
        FN = accPrun[4]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = (2 * (recall * precision)) / (recall + precision)
        avgPrecisionPrun = avgPrecisionPrun + precision
        avgRecallPrun = avgRecallPrun + recall
        avgF1ScorePrun = avgF1ScorePrun + f1score
        print(acc[0])
        avgAccPrun = accPrun[0] + avgAccPrun
        print()

        start_time = datetime.datetime.now()
        # for each sample in test Dat

    print("")
    print("")
    print("////////////////////////")
    print("Avarage Acc = " + str(avgAcc / 5))
    print("Avarage Precision = " + str(avgPrecision/5))
    print("Avarage Recall = " + str(avgRecall/5))
    print("Avarage F1Score = " + str(avgF1Score/5))
    print("Average Run Time = " + str(avgCompileTime / 5))
    print("////////////////////////")
    print("Avarage Acc Prun = " + str(avgAccPrun / 5))
    print("Avarage Precision Prun= " + str(avgPrecisionPrun / 5))
    print("Avarage Recall Prun= " + str(avgRecallPrun / 5))
    print("Avarage F1Score Prun= " + str(avgF1ScorePrun / 5))
    print("Average Run Time Prun= " + str(avgCompileTimePrun / 5))
    print("////////////////////////")
    print("")
    print("")