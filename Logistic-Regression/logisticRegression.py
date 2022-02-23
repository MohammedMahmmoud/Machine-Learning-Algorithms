import numpy as np
import pandas as pd


class LogisticRegression:
    learningRate = 0.5
    epochs = 150

    def __init__(self, features, target, data_size, features_number):
        self.feature = normalization(features, features_number)
        self.feature = np.array(self.feature)  # convert to array
        self.target = np.array(target).flatten()
        self.dataSize = data_size
        self.featuresNumber = features_number
        self.theta = [0] * features_number

    def update_learningRate(self, newLearningRate):
        self.learningRate = newLearningRate

    def update_epochs(self, newEpochs):
        self.epochs = newEpochs

    def y_predict(self):
        z = np.dot(self.feature, self.theta)
        return 1 / (1 + np.exp(-z))

    def costFunction(self):
        sum = np.sum(np.dot(np.log(self.y_predict()), self.target.T) + np.dot(np.log(1 - self.y_predict()), (1-self.target.T)))
        return (-1 / self.dataSize) * sum

    def gerdientDescent(self):
        # cost history and accuracy history of each epoch
        costHistory = [0] * self.epochs
        accuracyHistory = [0.0] * self.epochs

        for i in range(0, self.epochs):
            hypothesis = self.y_predict()
            correctCount = 0
            for j in range(self.dataSize):
                if hypothesis[j] == self.target[j]:
                    correctCount += 1

            gradiantEquation = self.feature.T.dot(self.target - hypothesis)

            # update theta
            self.theta = self.theta + (self.learningRate * gradiantEquation)

            costHistory[i] = self.costFunction()
            accuracyHistory[i] = correctCount / self.dataSize

        return self.theta

    def test_prediction(self, testFeature, testTarget, testDataSize):
        testFeatures = normalization(testFeature, self.featuresNumber)
        testFeatures = np.array(testFeatures)
        correctPrediction = 0

        testTargets = np.array(testTarget).flatten()
        for i in range(testDataSize):
            z = np.dot(testFeatures[i], self.theta)
            pred = 1 / (1 + np.exp(-z))
            if pred < 0.5:
                pred = 0
            else:
                pred = 1

            answer = "wrong"
            if pred == testTargets[i]:
                correctPrediction += 1
                answer = "correct"

            print("predicted: ", pred, "| actual: ", testTargets[i], '| and that prediction is : ', answer)
        print("number of correct predictions done is : ", correctPrediction)
        print("number of correct predictions done is : ", testDataSize - correctPrediction)
        print("accuracy : ", correctPrediction / testDataSize)


def normalization(feature, features_number):
    pd.options.mode.chained_assignment = None

    for i in range(0, features_number - 1):
        min = feature.iloc[:, i].min()
        max = feature.iloc[:, i].max()
        feature.iloc[:, i] = (feature.iloc[:, i] - min) / (max - min)
    return feature
