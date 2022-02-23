import pandas as pd
from pconst import const
from logisticRegression import LogisticRegression

# read dataset

df = pd.read_csv("heart.csv", index_col=0)
df.insert(0, 'ones', 1)

const.dataSize = len(df[['ones']])
const.trainingDataSize = int(const.dataSize * 0.8)

# define which feature we take in count and get their data
features = df[['trestbps', 'chol', 'thalach', 'oldpeak', 'ones']]
target = df[['target']]

# splitting data into train and test data
trainingFeature = features[:const.trainingDataSize]
trainingTarget = target[:const.trainingDataSize]

testingFeature = features[const.trainingDataSize:]
testingTarget = target[const.trainingDataSize:]

# defining the logistic regression object
logisticModel = LogisticRegression(trainingFeature, trainingTarget, const.trainingDataSize, features.shape[1])
theta = logisticModel.gerdientDescent()
print("model theta is : ", theta)


print("\n--------------------------------------------------------------------------------")
print("applying test data to the model : ")
logisticModel.test_prediction(testingFeature, testingTarget, testingTarget.size)
