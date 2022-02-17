import pandas as pd
from LinerRegressions import LinerRegressions
from pconst import const


#read dataset
df = pd.read_csv("house_data.csv", index_col=0)
df.insert(0, 'ones', 1)

const.dataSize = len(df[['ones']])
const.trainingDataSize = int(const.dataSize * 0.8)
#const.testingDataSize = int(const.dataSize * 0.2)

x = df[['grade', 'bathrooms', 'lat', 'sqft_living', 'view', 'ones']]
sqft_living = df[["sqft_living","ones"]]
sqft_living_plotting = df[["sqft_living"]]

y = df[['price']]


X_train = x[:const.trainingDataSize]
X_train_sqft = sqft_living[:const.trainingDataSize]
Y_train = y[:const.trainingDataSize]


X_test = x[const.trainingDataSize:]
X_sqft_test = sqft_living[const.trainingDataSize:]
Y_test = y[const.trainingDataSize:]

lr_singleVariable = LinerRegressions(const.trainingDataSize, 2, X_train_sqft, Y_train)
lr_multiVariable = LinerRegressions(const.trainingDataSize, 6, X_train, Y_train)

print(lr_singleVariable.gerdientDescent())

# print(lr_singleVariable.tetAccuracy(X_sqft_test, Y_test, const.dataSize-const.trainingDataSize))

lr_singleVariable.prediciton(X_sqft_test, Y_test, const.dataSize-const.trainingDataSize)

lr_singleVariable.plot(sqft_living_plotting, y)

print('*************************************************************************************')

print(lr_multiVariable.gerdientDescent())

# print(lr_multiVariable.tetAccuracy(X_test, Y_test, const.dataSize-const.trainingDataSize))

lr_multiVariable.prediciton(X_test, Y_test, const.dataSize-const.trainingDataSize)