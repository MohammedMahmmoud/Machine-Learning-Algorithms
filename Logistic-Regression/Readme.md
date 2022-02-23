# Logistic Regression form scratch without using sklearn library

## About main.py file 
* It's the main function
* It read the data file and split it into 80% train and 20% test.
* Features to take in count ar (trestbps, chol, thalach, oldpeak).
* Lastly it calls the functions of the object of the class "LogisticRegression" to run the algorithm.

## About LogisticRegression class
* Firstly it defines the number of epochs to (150) and learning rate to (0.5).
* (update_learningRate) and (update_epochs function) are used to set learning rate and epochs to the new entered value.
* At initialization we normalize the data we have using Min-Max Scaling to have the data at close scale.
* Now we are ready to call gerdientDescent function to set our model.
* After setting the model we can use (test_prediction) function to predict new values from the test data set and calculate the accuracy.

## About data file (heart.csv)
* it has 14 column(feature) and 304 row(data record).
