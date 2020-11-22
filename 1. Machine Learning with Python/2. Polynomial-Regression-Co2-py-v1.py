import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl

# read the csv file
dataset = pd.read_csv("1_Original_1995_1999_Fuel_Consumption_Ratings.csv", encoding='cp1252')
# take a look at dataset
#print(dataset.head())

# get the interested data
data = dataset[['MODEL','ENGINE_SIZE','CYLINDERS', 'FUEL_CONSUMPTION', 'CO2']]
# to save some time, just analyse first 2000 data
data = data.head(2000)

# first things to do: inspect visually to see the data, whether it is linear, non-linear, polynomial relative
#plt.scatter(data.FUEL_CONSUMPTION, data.CO2, color='blue')
#plt.show()
#plt.scatter(data.ENGINE_SIZE, data.CO2, color='blue')
#plt.show()
# we now know that fuel_consumption is more linear and engine_size is relativ polynomial

# ======================= so let's construct the different dype of ML into these two data========================================

# fist to seperate the data into train and test set
mask = np.random.rand(len(data)) < 0.8
print(mask)
train = data[mask]
test = data[~mask]

# consider the module
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

regr = linear_model.LinearRegression()
poly = PolynomialFeatures(3)
# PolynomialFeatures() function in Scikit-learn library, drives a new feature sets
# from the original feature set. That is, a matrix will be generated consisting of all
# polynomial combinations of the features with degree less than or equal to the specified degree.
# For example, lets say the original feature set has only one feature, ENGINESIZE.
# Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:

# =======================================================================================================================
# Let's start with Linear Regression Model from Fuel_consumption
train_fuel_x = np.asanyarray(train[['FUEL_CONSUMPTION']])
train_fuel_y = np.asanyarray(train[['CO2']])
# fit into the model
regr.fit(train_fuel_x, train_fuel_y)
print('Coefficient: {}'.format(regr.coef_))  # pay attention to the data type of the coefficient
print('Intercept: {}'.format(regr.intercept_))

# Let's plot the scatter and model together to see how it predict
plt.scatter(data.FUEL_CONSUMPTION, data.CO2, color='blue')
plt.xlabel('Fuel Consumption')
plt.ylabel('CO2 Emission')
# plot out the predicted line
plt.plot(train_fuel_x, regr.intercept_[0]+regr.coef_[0][0]*train_fuel_x, '-r')
plt.show()

# start Evaluation
test_fuel_x = np.asanyarray(test[['FUEL_CONSUMPTION']])
test_fuel_y = np.asanyarray(test[['CO2']])
test_fuel_predict = regr.predict(test_fuel_x)
# test_fuel_predict = regr.intercept_[0]+regr.coef_[0][0]*test_fuel_x

print("Mean absolute error: {:.2f}".format(np.mean(np.absolute(test_fuel_predict - test_fuel_y))))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_fuel_predict - test_fuel_y) ** 2))
print("R2-score: %.2f" % r2_score(test_fuel_y, test_fuel_predict))

# ========================================================================================================================
# Let's start with Polynomial Regression Model from Engine_size
train_engine_x = np.asanyarray(train[['ENGINE_SIZE']])
train_engine_y = np.asanyarray(train[['CO2']])
test_engine_x = np.asanyarray(test[['ENGINE_SIZE']])
test_engine_y = np.asanyarray(test[['CO2']])

# train the Polynomial Regression Model
# fit_transform takes our x values, and output a list of our data raised from power of 0 to power of 3
# (since we set the degree of our polynomial to 3). from poly = PolynomialFeatures(3)
train_engine_x_poly = poly.fit_transform(train_engine_x)  # !!!!!!!!!!!!!!!!!!!

# remember the basic polynomial regression formula: y_hat = (intercept)+(coefficient[1])*x + (coefficient[2])*x^2 + (coefficient[3])*x^3
# which can be transformed to linear Regression y_hat = (intercept)+(coefficient[1])*x1 + (coefficient[2])*x2 + (coefficient[3])*x3
regr.fit(train_engine_x_poly, train_engine_y)  # same as building the multi Regression model
                                               # train_engine_x_ploy is an len(data)*4 array
print('Coefficients: ', regr.coef_)
print('Intercept: {}'.format(regr.intercept_))

# Let's plot the scatter and model together to see how it predict
plt.scatter(train.ENGINE_SIZE, train.CO2, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emission')

# plot out the line correct way by using train data :
#   yy = regr.intercept_[0]+ regr.coef_[0][1]*train_engine_x+ regr.coef_[0][2]*np.power(train_engine_x, 2)+ regr.coef_[0][3]*np.power(train_engine_x, 3)
#   plt.plot(train_engine_x,YY, '-r')
# but in oder to just create a perfect line, I just show it with 0-10 value
XX = np.arange(0.0, 10.0, 0.1)
YY = regr.intercept_[0]+regr.coef_[0][1]*XX + regr.coef_[0][2]*np.power(XX, 2) + regr.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, YY, '-r')
# plt.show()

# start Evaluation
test_engine_x_poly = poly.fit_transform(test_engine_x)
print(test_engine_x_poly)
test_engine_predict = regr.predict(test_engine_x_poly)
print(test_engine_predict)
# show the test prediction line
plt.plot(test_engine_x, test_engine_predict, 'og')
plt.show()

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_engine_predict - test_engine_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_engine_predict - test_engine_y) ** 2))
print("R2-score: %.2f" % r2_score(test_engine_predict, test_engine_y))






