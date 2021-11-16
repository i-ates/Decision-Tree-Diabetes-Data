import numpy as np

def discreteFeature(df, featureIndexes, bin):
    # find min max value of feature at featureIndex
    for z in range(0, len(featureIndexes)):
        minVal = df[featureIndexes[z]][0]
        maxVal = df[featureIndexes[z]][0]
        for i in range(0, df.shape[1]):
            if i == featureIndexes[z]:
                for j in range(0, df.shape[0]):
                    if df[j][i] > maxVal:
                        maxVal = df[j][i]
                    if df[j][i] < minVal:
                        minVal = df[j][i]
        # find bigness of each part of continues values
        meanVal = (maxVal - minVal) / bin

        # transform continues values to discrete values
        for i in range(0, df.shape[1]):
            if i == featureIndexes[z]:
                for j in range(0, df.shape[0]):
                    df[j][i] = int((df[j][i] - minVal) // meanVal)

    return df

def mapDataFrame(df, indexList):
    newDf = np.zeros(shape=(df.shape[0], df.shape[1]), dtype=int)

    for i in range(0, df.shape[1]):
        if i in indexList:
            elementList = []
            for j in range(0, df.shape[0]):
                if df[j][i] not in elementList:
                    elementList.append(df[j][i])
            for j in range(0, df.shape[0]):
                for z in range(0, len(elementList)):
                    if elementList[z] == df[j][i]:
                        newDf[j][i] = z
                        break
            elementList.sort()
            print("***index : "+ str(i)+"****")
            for x in range(0 , len(elementList)):
                print(elementList[x] + " : " + str(x))
        else:
            for j in range(0, df.shape[0]):
                newDf[j][i] = df[j][i]


    return newDf