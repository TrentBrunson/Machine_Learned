#%%
# Build a loan approver using the SVM algorithm 
# and compare the accuracy and performance of the SVM model 
# with the Logistic Regression model.

# 1. split  dataset 
# 2. create and train a model
# 3. create predictions
# 4. validate the model
from pathlib import Path
import numpy as np
import pandas as pd
# %%
# Read in the data
# Note: The following data has been normalized between 0 and 1
data = Path('../Resources/loans.csv')
df = pd.read_csv(data)
df.head()
# %%
# Separate the Features (X) from the Target (y)
# Segment the features from the target
y = df["status"]
X = df.drop(columns="status")
# %%
# Split our data into training and testing
# Use the train_test_split function to create training and testing subsets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)
X_train.shape
# %%
# Create a SVM Model
# Instantiate a linear SVM model
from sklearn.svm import SVC
model = SVC(kernel='linear')
# %%
# Fit (train) or model using the training data
model.fit(X_train, y_train)
# %%
# Score the model using the test data
#%%
# Make predictions using the test data
y_pred = model.predict(X_test)
results = pd.DataFrame({
    "Prediction": y_pred, 
    "Actual": y_test
}).reset_index(drop=True)
results.head()
# %%
# validate model with accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
# %%
# generate confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
# %%
# generate classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# %%
# asses the performance of the logistic regression model 
# and compare this to the SVM predictions
