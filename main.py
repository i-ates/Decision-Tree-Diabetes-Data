import csv

import pandas as pd  # just uses for read csv
import numpy as np  # numpy library used throughout the whole project

# part1
import dataPreparation
import kFoldCrossValidationDecisionTree

dfDiabetes = pd.read_csv("diabetes_data_upload.csv")

npDiabetes = dfDiabetes.to_numpy()
npDiabetes = dataPreparation.discreteFeature(npDiabetes, [0], 5)

np.random.shuffle(npDiabetes)
npDiabetes = dataPreparation.mapDataFrame(npDiabetes, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

pdDibates = pd.DataFrame(npDiabetes, columns=dfDiabetes.columns.values)

pdDibates.to_csv("x.csv", index= False)

out = []
csvfile = open("x.csv", 'r')
fileToRead = csv.reader(csvfile)

headers = next(fileToRead)

# iterate through rows of actual data
for row in fileToRead:
    out.append(dict(zip(headers, row)))

kFoldCrossValidationDecisionTree.kFoldCrossValidationClasification(out)
