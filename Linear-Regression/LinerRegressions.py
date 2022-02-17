import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pconst import const


class LinerRegressions:
    const.iterationNumber = 100000
    const.learningRate = 0.00000005
    # const.learningRate = 0.5
    # const.learningRate = 0.00000000000001
    # const.learningRate = 0.00005
    # const.learningRate = 1

    def __init__(self, dataSize, featureNum, x, y):
        self.dataSize = dataSize
        self.featureNum = featureNum
        self.x = x
        self.actualOutput = np.array(y).flatten()  # convert to array
        self.theta = [0]
        for i in range(0, self.featureNum - 1):
            self.theta += [0]

        # self.theta = np.array(self.theta)

        self.normalization()

        self.x = np.array(self.x)  # convert to array

    def normalization(self):
        pd.options.mode.chained_assignment = None

        for i in range(0, self.featureNum - 1):
            min = self.x.iloc[:, i].min()
            max = self.x.iloc[:, i].max()
            self.x.iloc[:, i] = (self.x.iloc[:, i] - min) / (max - min)

    def costFunction(self):
        costValue = np.sum((self.x.dot(self.theta) - self.actualOutput) ** 2) / (2 * self.dataSize)
        return costValue

    def gerdientDescent(self):
        costHistory = [0] * const.iterationNumber
        error = 0

        for i in range(0, const.iterationNumber):
            x_dot_theta = self.x.dot(self.theta)

            gradiantEquation = self.x.T.dot(x_dot_theta - self.actualOutput) / self.dataSize

            self.theta = self.theta - (const.learningRate * gradiantEquation)

            costHistory[i] = self.costFunction()
            error += costHistory[i]

        return costHistory, self.theta

    def tetAccuracy(self, testX, testY, testDataSize):
        testX = np.array(testX)
        testY = np.array(testY)
        testDataSize = np.array(testDataSize)
        x = (np.sum(testX.dot(self.theta) / testY)) / testDataSize
        return x

    def prediciton(self, test, price_test, testDataSize):
        test = np.array(test)
        price_test = np.array(price_test).flatten()
        for i in range(testDataSize):
            sum=0
            for j in range(self.featureNum):
                sum += self.theta[j] * test[i][j]

            print("predicted: ", sum, ", actual: ", price_test[i], ' differance: ', price_test[i] - sum)

    def plot(self, sqft_living_train2, y):
        print(sqft_living_train2.shape)
        bestFitX = np.linspace(0, 10000, 10000)

        bestFitY = [self.theta[1] + self.theta[0] * newX for newX in bestFitX]  # plot regression line across data points
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Real vs predicted values")
        plt.plot(sqft_living_train2, y, '.')
        plt.plot(bestFitX, bestFitY, '-')

        plt.show()
