import numpy as np
import math


# decision Tree

# calculate the splitting point of continuous feature
def calculateSplittingPoint(feature):
    res = []
    sortedFea = sorted(feature)
    for i in range(1, len(sortedFea)):
        if sortedFea[i] == sortedFea[i - 1]:
            continue
        else:
            middle = (sortedFea[i] + sortedFea[i - 1]) / 2
            res.append(middle)
    return res


# calculate information entropy
def calculateEntropy(y):
    entropy = 0
    uniqueY = np.unique(y)
    for label in uniqueY:
        num = len(y[y == label])
        p = num / len(y)
        entropy += (-p) * math.log2(p)
    return entropy


def calculateInfoGain(dataset, y, feature):
    splitPoints = calculateSplittingPoint(dataset[:, feature])
    # print(split_points)
    gainList = []  # pairs [gain,splitting point]
    for p in splitPoints:
        # <=,the value of feature is less than p
        lessIndex = np.where(dataset[:, feature] <= p)
        # lessindex : ([1,5,9,10])
        E1 = calculateEntropy(y[lessIndex])
        # >
        greatIndex = np.where(dataset[:, feature] > p)
        E2 = calculateEntropy(y[greatIndex])
        # dataset entropy
        E = calculateEntropy(y)
        D1 = len(lessIndex[0])
        D2 = len(greatIndex[0])
        D = len(dataset)

        gain = E - (D1 / D) * E1 - (D2 / D) * E2
        gainList.append([round(gain, 3), p])

    gainList.sort(key=lambda x: x[0], reverse=-1)
    # if we can not split,we return -1,-1
    if len(gainList) == 0:
        return [-1, -1]
    return gainList[0]


class DecisionTree:
    def __init__(self, eta, feature):
        self.eta = eta
        self.feature = feature

    def fit(self, totalX, totalY, prun=False):
        xTrain, yTrain, xValid, yValid = totalX[:-100], totalY[:-100], totalX[-100:], totalY[-100:]
        self.prun = prun
        self.validX = xValid
        self.validY = yValid
        self.numsFea = xTrain.shape[1]
        self.numsSam = xTrain.shape[0]
        self.root = dict()  # this is the decision tree
        self.graph = dict()  # this is convenient for visualizing the tree
        self.maxdepth = self.buildTree(xTrain, yTrain, 0, self.graph, self.root)


    def vote(self, y):
        mostCommon = None
        maxCount = 0
        for label in np.unique(y):
            # choose the majority as label
            count = len(y[y == label])
            if count > maxCount:
                mostCommon = label
                maxCount = count
        return mostCommon

    def predict(self, X):
        y = []
        for x in X:
            dic = self.root
            while True:
                if 'leaf' in dic.keys():
                    y.append(dic['leaf'])
                    break
                if float(x[dic['feature']]) <= dic['val']:
                    dic = dic['left']
                else:
                    dic = dic['right']
        return np.array(y)

    def accuracy(self, X, y):
        pred = self.predict(X)
        acc = (pred == y).sum() / y.shape[0]
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(0, len(pred)):
            if pred[i] == 1:
                if pred[i] == y[i]:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                if pred[i] == y[i]:
                    TN = TN + 1
                else:
                    FN = FN + 1
        return [acc, TP, TN, FP, FN]

    def pruning(self, X, y, root):
        # is_prun
        isPrun = False

        # after pruning,calculate the accuracy on train set
        afterPruVoteY = self.vote(y)
        root['leaf'] = np.asscalar(afterPruVoteY)
        accAfterPru = self.accuracy(self.validX, self.validY)
        # recover the original
        root.pop('leaf')
        # non pruning,calculate the accuracy on valid set
        bestGain, bestF, bestPoint = self.splittingPoint(X, y)
        if bestF == -1 or bestGain < self.eta:  # there is no feature that could be divided or less than eta
            isPrun = True
            return isPrun

        root['val'] = bestPoint
        root['feature'] = bestF
        leftIndex = np.where(X[:, bestF] <= bestPoint)
        rightIndex = np.where(X[:, bestF] > bestPoint)
        yleft = y[leftIndex]
        voteYLeft = self.vote(yleft)
        yright = y[rightIndex]
        voteYRigt = self.vote(yright)
        root['left'] = dict()
        root['right'] = dict()
        root['left']['leaf'] = np.asscalar(voteYLeft)
        root['right']['leaf'] = np.asscalar(voteYRigt)
        accBeforePru = self.accuracy(self.validX, self.validY)

        # recover the original
        root.pop('left')
        root.pop('right')
        root.pop('val')
        root.pop('feature')

        # if pruning do make acc greater,return True
        if accAfterPru >= accBeforePru:
            isPrun = True
        return isPrun

    def splittingPoint(self, X, y):
        gainList = []
        # calculate gain for each features and different splitting points
        for f in range(self.numsFea):
            gain, point = calculateInfoGain(X, y, f)
            gainList.append([gain, f, point])
        gainList.sort(key=lambda x: x[0], reverse=-1)
        bestGain, bestF, bestPoint = gainList[0]
        return bestGain, bestF, bestPoint

    def buildTree(self, X, y, depth, graph=None, root=None):
        if len(np.unique(y)) == len(y):
            # return the label at first index
            root['leaf'] = np.asscalar(y[0])
            graph['label:' + str(np.asscalar(y[0]))] = 'leaf'
            return depth + 1
        else:
            if self.prun == True:
                isPrun = self.pruning(X, y, root)
            else:
                # don't do pruning
                isPrun = False
            if isPrun:
                beforePruVoteY = self.vote(y)
                root['leaf'] = np.asscalar(beforePruVoteY)
                graph['label:' + str(np.asscalar(beforePruVoteY))] = 'leaf'
                return depth + 1
            else:
                bestGain, bestF, bestPoint = self.splittingPoint(X, y)
                if bestF == -1 or bestGain < self.eta:  # there is no feature that could be divided or less than eta
                    # return the majority
                    voteY = self.vote(y)
                    root['leaf'] = np.asscalar(voteY)
                    graph['label:' + str(np.asscalar(voteY))] = 'leaf'
                    return depth + 1

                key = self.feature[bestF] + "<=" + str(bestPoint)
                # graph painting dict
                graph[key] = dict()
                less = dict()
                great = dict()
                graph[key]['yes'] = less
                graph[key]['no'] = great

                # tree structure
                root['val'] = bestPoint
                root['majority'] = np.asscalar(self.vote(y))
                root['feature'] = bestF
                leftIndex = np.where(X[:, bestF] <= bestPoint)
                rightIndex = np.where(X[:, bestF] > bestPoint)

                left = dict()
                root['left'] = left
                root['right'] = dict()
                root['right']['leaf'] = np.asscalar(
                    self.vote(y[rightIndex]))  # the right tree choose the majority as label
                ldepth = self.buildTree(X[leftIndex], y[leftIndex], depth + 1, less, left)

                root.pop('right')
                right = dict()
                root['right'] = right
                rdepth = self.buildTree(X[rightIndex], y[rightIndex], depth + 1, great, right)
                return max(ldepth, rdepth)
