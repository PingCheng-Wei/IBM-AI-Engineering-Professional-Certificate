import numpy as np
import matplotlib as plt
import pandas as pd

# read csv file as dataframe
df = pd.read_csv('5_drug200.csv')
print(df.head(10))
print(df.describe())

# get the value out of dataframe and separate to feature matrix and response vector
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df[['Drug']].values

# we have to convert those features to numerical values.
# pandas.get_dummies() Convert categorical variable into dummy/indicator variables.
from sklearn import preprocessing  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
le_sex = preprocessing.LabelEncoder().fit(['F', 'M'])
x[:, 1] = le_sex.transform(x[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
x[:, 2] = le_BP.transform(x[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
x[:, 3] = le_Chol.transform(x[:, 3])

print(x[0:5])

# ==================================== Setting up the Decision Tree =============================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=3)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

# set up model
from sklearn.tree import DecisionTreeClassifier
drugtree = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(train_x, train_y)
print(drugtree)

# make prediction
tree_predict = drugtree.predict(test_x)
print(tree_predict)

# Evaluation
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(test_y, tree_predict))

# Visualization:
# go to see example code in the notebook  !!!!





