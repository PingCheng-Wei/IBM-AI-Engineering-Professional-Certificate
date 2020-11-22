import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("3_china_gdp.csv")
print(dataset.head())

# take a first look at dataset
plt.scatter(dataset["Year"], dataset["Value"], color='blue', label='data')
plt.xlabel("Year 1960 - 2014")
plt.ylabel("GDP Value")
plt.show()

# ==========================================================================
# after first look, now give it a guess for module
# let's guess sigmoid & split the dataset into test and train set

# define sigmoid function
def sigmoid(x, beta1, beta2):
    y = 1 / (1 + np.exp(-beta1*(x - beta2)))
    return y
# example of the guess module by whatever beta1 beta2
beta_1 = 0.10
beta_2 = 1990.0
# logistic function
Y_pred = sigmoid(dataset['Year'], beta_1, beta_2)
# plot initial prediction against datapoints
plt.plot(dataset['Year'], Y_pred*15000000000000., label='predict line')
plt.plot(dataset['Year'], dataset['Value'], 'ro', label='true line')
plt.show()

# better to normalize the data and then analyse before split the data
data_x_norm = dataset['Year'] / max(dataset['Year'])
data_y_norm = dataset['Value'] / max(dataset['Value'])

# split the data
msk = np.random.rand(len(dataset)) < 0.8
train_x = data_x_norm[msk]
train_y = data_y_norm[msk]
test_x = data_x_norm[~msk]
test_y = data_y_norm[~msk]

# find a good beta1 & beta2 to fit the model
# one way to achieve it is by using curve_fit function from optimize of scipy
from scipy.optimize import curve_fit  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
popt, pcov = curve_fit(sigmoid, train_x, train_y)
# print out final parameter
print("Beta 1: {}, Beta 2: {}".format(popt[0], popt[1]))
print("Check out what popt is : ", popt)
print('Check out what pcov is: {}'.format(pcov))

# get the model right with the final Parameter
test_predict = sigmoid(test_x, popt[0], popt[1])
plt.plot(test_x, test_y, '-b', label='True Line')
plt.plot(test_x, test_predict, 'r--', label='Predictive Line')
plt.legend(loc='best')  # definitely need this to show out the label box
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Evaluation
print(" Mean absolute error: {}".format(np.mean(np.absolute(test_y - test_predict))))
print(" Residual sum of squares: {}".format(np.mean((test_y - test_predict)**2)))
from sklearn.metrics import r2_score
print(" R2 score: {}".format(r2_score(test_y, test_predict)))
