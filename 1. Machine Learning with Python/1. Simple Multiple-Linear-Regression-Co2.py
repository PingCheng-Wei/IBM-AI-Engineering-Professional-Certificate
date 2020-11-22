import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# check the current position
import os
# print (os.getcwd())
# print (os.listdir())


# read the csv file in current directory
dataset = pd.read_csv("1_Original_1995_1999_Fuel_Consumption_Ratings.csv", encoding='cp1252')
# another way to read csv by using panda
# import csv
# Data = csv.reader(open("Original_1995_1999_Fuel_Consumption_Ratings.csv"))

# take a look at the dataset
#print (dataset.head()) ##############
# summarize the data
#print (dataset.describe())  ############
# get interested and relevant data from dataset
data = dataset[['ENGINE_SIZE', 'CYLINDERS', 'FUEL_CONSUMPTION', 'CO2']]
# in oder to avoid too big input data size and too long programm analysis running time
# just get the first 1000 Data from Dataset
data = data.head(2000)


# plot each of these features
#data.hist() ########
#plt.show()  ########

# plot each of these features vs the Emission, to see how linear is their relation:
#plt.scatter(data.FUEL_CONSUMPTION, data.CO2, color='blue')
plt.xlabel("FUEL_CONSUMPTION")
plt.ylabel("CO2_Emission")
#plt.show()

#plt.scatter(data.ENGINE_SIZE, data.CO2, color='red')
plt.xlabel("Engine size")
plt.ylabel("CO2_Emission")
#plt.show()

############################### Start from here #################################
# Let's go with Linear Regression model
# Creating train and test dataset

# so first:
# split our dataset intotrain and test sets, 80% of the entire data for training, and the 20% for testing.
# We create a mask to select random rows using np.random.rand() function:
mask = np.random.rand(len(data)) < 0.8   # !!!!!!!!!!!!!!!!!!!!!!!!
print(mask)
train = data[mask]
test = data[~mask]

# import Linear Regression model
from sklearn import linear_model
regr = linear_model.LinearRegression()
# convert the column (list) into array data structure
train_x = np.asanyarray(train[["ENGINE_SIZE"]])  # almost same as np.array
train_y = np.asanyarray(train[["CO2"]])
# fit into the module => Output: regr.coefficient, regr.interception
regr.fit(train_x, train_y)
# remember the basic linear formel: y = (coefficient)*x + (interception) !!!!!!
print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)

# Plot outputs !!!!!!!!!!!!
plt.scatter(train.ENGINE_SIZE, train.CO2,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') # !!!!!!!!!!!!!!!!!!!!
plt.xlabel("ENGINE_SIZE")
plt.ylabel("Emission")
plt.show()

############ Evaluation #################3
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINE_SIZE']])
test_y = np.asanyarray(test[['CO2']])
test_y_predict = regr.predict(test_x)

print("Mean absolute error: {:.2f}".format(np.mean(np.absolute(test_y_predict - test_y))))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_predict - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_predict))




#################################################################################
# ++++++++++++++++++++++++ Multiple Linear Regression +++++++++++++++++++++++++
# acutally almost same as Simple Linear Regression

msk = np.random.rand(len(data)) < 0.8
mul_train = data[msk]
mul_test = data[~msk]

mul_train_x = np.asanyarray(mul_train[["ENGINE_SIZE", 'CYLINDERS', 'FUEL_CONSUMPTION']])
mul_train_y = np.asanyarray(mul_train[["CO2"]])
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(mul_train_x, mul_train_y)

# remember the basic linear formel: y = (interception) + (coefficient[0])*x +(coefficient[1])*x^2 + (coefficient[2])*x^3  !!!!!!
print("Coefficient: {}".format(regr.coef_))
print("Intercept: {}".format(regr.intercept_))

######## Prediction ###########
mul_test_x = np.asanyarray(mul_test[["ENGINE_SIZE", 'CYLINDERS', 'FUEL_CONSUMPTION']])
mul_test_y = np.asanyarray(mul_test[["CO2"]])
mul_test_predict = regr.predict(mul_test_x)

print("Residual sum of squares: %.3f" % np.mean((mul_test_predict - mul_test_y) ** 2))
print("Variance score: {:.3f}".format(regr.score(mul_test_x, mul_test_y)))  ############## !!!!!! Difference !!!!!!
