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
# %%
# Create a Logistic Regression Model
# set upper limit on # of iterations to 200
# see random_state to 1 for result duplication purposes
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',
                                max_iter=200,
                                random_state=1)
# %%
# 2. train/fit model using the training data
classifier.fit(X_train, y_train)
# %%
# make predictions
y_pred = classifier.predict(X_test)
results = pd.DataFrame({"Prediction": y_pred, "Actual": y_test}).reset_index(drop=True)
results.head(20)

# %%
# 4. validate model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
# %%
# show matrix of positives, negatives, false positives, & false negatives
from sklearn.metrics import confusion_matrix, classification_report
    # 	            Predicted True	Predicted False
# Actually True	    TRUE POSITIVE	FALSE NEGATIVE
# Actually False    FALSE POSITIVE	TRUE NEGATIVE

# Precision = TP/(TP + FP)
# Sensitivity (recall) = TP/(TP + FN)

# F1 score (harmonic mean) = 2(Precision * Sensitivity)/(Precision + Sensitivity)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
# %%
# calculate precision, sensitivity (recall) and
# F1 ration (harmonized mean)
report = classification_report(y_test, y_pred)
print(report)
# %%
# log model same as SVM when recognizing conditions to deny; 
# horrible at approvals; accuracy 52% = coin toss
# SVM model better at recommending approvals
# SVM only had 60% accuracy, not much better than a coin toss