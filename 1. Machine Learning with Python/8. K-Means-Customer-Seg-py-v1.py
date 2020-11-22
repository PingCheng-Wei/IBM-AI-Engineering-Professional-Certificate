import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# ============ k-Means on a randomly generated dataset ==================

# First we need to set up a random seed. Use numpy's random.seed() function, where the seed will be set to 0
np.random.seed(0)

# Next we will be making random clusters of points by using the make_blobs class.
# The make_blobs class can take in many inputs, but we will be using these specific ones.
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2,-3], [1,1]], cluster_std=0.9)

plt.scatter(X[:,0], X[:,1], marker='.')
plt.show()

# Setting up K-Means
from sklearn.cluster import KMeans
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
# fit the data into modul
k_means.fit(X)
k_means_labels = k_means.labels_
print(k_means_labels)
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)

# Creating the visual plot !!!!
# see the algorithm in Jupyter Notebook file !!!!

# =========================================================================
# ====================Customer Segmentation with K-Means===================
cust_df = pd.read_csv("8_Cust_Segmentation.csv")
print(cust_df.head())

# Pre-processing
# Address in this dataset is a categorical variable.
# k-means algorithm isn't directly applicable to categorical variables
# because Euclidean distance function isn't really meaningful for discrete variables.
# So, lets drop this feature and run clustering.
df = cust_df.drop('Address', axis=1)
print(df.head())

# Normalizing over the standard deviation
from sklearn.preprocessing import StandardScaler
x = df.values[:, 1:]
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
print(Clus_dataSet)

# Modeling
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(x)
labels = k_means.labels_
print(labels)

# Insights
df["Cluster_km"] = labels
print(df.head(5))
print(df.groupby('Cluster_km').mean())

# Visualization of the distrivution of customers based on their age and income
area = np.pi * (x[:, 1])**2
plt.scatter(x[:, 0], x[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

# Visualization of 3D modul !!!!!
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=(0, 0, .95, 1), elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(x[:, 1], x[:, 0], x[:, 3], c=labels.astype(np.float))
plt.show()
