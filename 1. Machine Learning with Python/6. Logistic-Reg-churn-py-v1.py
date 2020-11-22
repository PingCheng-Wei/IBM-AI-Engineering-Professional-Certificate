import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

churn_df = pd.read_csv('6_ChurnData.csv')
print(churn_df.head())

# Data pre-processing and selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

# separate the feature matrix and response vector
x = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless']])
y = np.asarray(churn_df[['churn']])

# normalize the data
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
print(x[0:5])

# split train test dataset
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=4)
print('Train set: {}, {}'.format(train_x.shape, train_y.shape))
print('Test set: {}, {}'.format(test_x.shape, test_y.shape))

# set up model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(train_x, train_y)
print(LR)

# prediction
y_predict = LR.predict(test_x)
print(y_predict)

# predict_proba returns estimates for all classes, ordered by the label of classes.
# So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
y_predict_prob = LR.predict_proba(test_x)
print(y_predict_prob)

# Evaluation
# jaccard index !!!!!!!!!!!!!!!!!!
# from sklearn.metrics import jaccard_similarity_score !!!!!!!!!!!
# jaccard_similarity_score(test_y, y_predict)  !!!!!!!!!!!!!!!!!


# from sklearn.metrics import log_loss
# log_loss(test_y, y_predict_prob)
