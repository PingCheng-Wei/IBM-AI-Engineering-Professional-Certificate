import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file in current directory and store it as data frame
df = pd.read_csv("4_teleCust1000t.csv")
print(df.head())

# let see how many of each class in our data
print(df['custcat'].value_counts())
# so we now know there are 4 different class

# Visualization of the data
plt.hist(df['income'], bins=100)  # bins decides how many column to show => higher = better resolution
# df.hist(column='income', bins=50)  # same way to do it
plt.show()

# take a look at all feature sets
print(df.columns)
# get the dataset from the feature that we are interested
dataset = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  #.astype(float)
print(dataset)
y = df['custcat'].values
print(y)

# ======================== start KNN algorithms =====================================
# Normalize the Data
from sklearn import preprocessing
dataset = preprocessing.StandardScaler().fit(dataset).transform(dataset.astype(float))
print(dataset)

# split train test data
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(dataset, y, test_size=0.2, random_state=4)
print('Train set:', train_x.shape,  train_y.shape)
print('Test set:', test_x.shape,  test_y.shape)

# Classification KNN
from sklearn.neighbors import KNeighborsClassifier
# define the model
k = 4  # k => how many neighbors should it examine in order to decide the class
neigh = KNeighborsClassifier(n_neighbors=k)
# fit the train_x and train_y into the model
neigh.fit(train_x, train_y)
print(neigh)

# make the prediction
y_predict = neigh.predict(test_x)
print(y_predict)

# Evaluation
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(train_y, neigh.predict(train_x)))
print("Test set Accuracy: ", metrics.accuracy_score(test_y, y_predict))

# lets automatic test through lots of different Ks
ks = 10
mean_acc = np.zeros((ks-1))
std_acc = np.zeros(ks-1)
ConfustionMx = []
for k in range(1, ks):
    # train model and Predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(train_x, train_y)
    # prediction
    y_predict = neigh.predict(test_x)
    mean_acc[k-1] = metrics.accuracy_score(test_y, y_predict)
    std_acc[k-1] = np.std(y_predict == test_y)/np.sqrt(y_predict.shape[0])

print("Mean Accuracy: ", mean_acc)
print("Standard deviation: ", std_acc)

# plot out the model accuracy for different number of neighbors
plt.plot(range(1, ks), mean_acc, 'g')
plt.fill_between(range(1, ks), mean_acc - 1*std_acc, mean_acc + 1*std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()



