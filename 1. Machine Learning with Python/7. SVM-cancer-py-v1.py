import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the csv file
cell_df = pd.read_csv('7_cell_samples.csv')
print(cell_df.head())

# special plot method !!!!!!!!!!!!!!!!!!!!!!!!!!
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
plt.show()

# take a look at columns data types
print(cell_df.dtypes)

# change the non-numerical columns into numerical columns
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors = 'coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)
print(cell_df)

# get the feature matrix and respond vector
x = np.asarray(cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
y = np.asarray(cell_df[['Class']])
print(x[0:5])
print(y[0:5])

# train test dataset
from sklearn.model_selection import train_test_split
train_x,test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=4)
print ('Train set:', train_x.shape,  train_y.shape)
print ('Test set:', test_x.shape,  test_y.shape)

# set up SVM model with sklearn
from sklearn import svm
svm_model = svm.SVC(kernel='rbf').fit(train_x, train_y)
print(svm_model)

# Prediction
y_predict = svm_model.predict(test_x)
print(y_predict)

# Evaluation
from sklearn.metrics import f1_score
f1_score(test_y, y_predict, average='weighted')
print("f1 score accuracy: {}".format(f1_score(test_y, y_predict, average='weighted')))

# from sklearn.metrics import jaccard_similarity_score
# jaccard_similarity_score(test_y, y_predict)
# print("f1 score accuracy: {}".format(jaccard_similarity_score(test_y, y_predict)))
