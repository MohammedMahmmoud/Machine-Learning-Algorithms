# Linear Regression work on single or multi variables without using sklearn library

## About file linear_main 
* It's the main function
* It read the data file and split it into 80% train and 20% test.
* When working on single value linear regression the feature to work on is house size (square feet).
* With multi variables it takes ('grade', 'bathrooms', 'lat', 'house size (square feet)', 'view') features in the data in counter.
* Lastly it calls the functions of the object of the class "LinerRegressions" to run the algorithm.

## About LinerRegressions class
* Firstly it defines the number of iterations to (100000) and learning rate to (0.00000005).
* At initialization we normalize the data we have using Min-Max Scaling to have the data at close scale.
* Now we are ready to call gerdientDescent function to set our model.
* After setting the model we can use prediciton function to predict new values from the test data and calculate the accuracy.
* plot function is used with single value linear regression only to plot our model above the existing data.

## About data file (house_data.csv)
* it has 21 column(feature) and 21614 row(data record).
