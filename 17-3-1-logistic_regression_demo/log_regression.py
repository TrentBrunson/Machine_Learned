#%%
import matplotlib.pyplot as plt
import pandas as pd
# %%
from sklearn.datasets import make_blobs
# centers argument specifies the number of clusters in the dataset; 
# in this case there are two clusters. 
# The random_state ensures reproducibility of this dataset
X, y = make_blobs(centers=2, random_state=42)

print(f"Labels: {y[:10]}")
print(f"Data: {X[:10]}")
# %%
plt.scatter(X[:, 0], X[:, 1], c=y)
# %%
# split the dataset into two: train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
    y, random_state=1, stratify=y)
# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', random_state=1)
classifier
# %%
