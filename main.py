import pandas as pd  # just uses for read csv
import numpy as np  # numpy library used throughout the whole project

# part1
import dataPreparation
import kFoldCrossValidationDecisionTree

dfDiabetes = pd.read_csv("diabetes_data_upload.csv")
dfDiabetes = dfDiabetes.to_numpy()

dfDiabetes = dataPreparation.discreteFeature(dfDiabetes, [0], 5)

np.random.shuffle(dfDiabetes)
dfDiabetes = dataPreparation.mapDataFrame(dfDiabetes, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
kFoldCrossValidationDecisionTree.kFoldCrossValidationClasification(dfDiabetes)
