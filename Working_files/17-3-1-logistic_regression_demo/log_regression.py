#%%
import matplotlib.pyplot as plt
import pandas as pd

# steps of log regression (classification) model
# 1. Create a model with LogisticRegression().
# 2. Train the model with model.fit().
# 3. Make predictions with model.predict().
# 4. Validate the model with accuracy_score().
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
# train the model
classifier.fit(X_train, y_train)
# %%
# validate the model and load into a new DF using predict() method
predictions = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions, "Actual": y_test})
# %%
# examine new dataframe from last cell and calculate an accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
# %%
# create a new red dot as a new data point for classification model

import numpy as np
new_data = np.array([[-2, 6]])
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(new_data[0, 0], new_data[0, 1], c="r", marker="o", s=100)
plt.show()
# %%
predictions = classifier.predict(new_data)
print("Classes are either 0 (purple) or 1 (yellow)")
print(f"The new point was classified as: {predictions}")
# %%
