# Undersampling
# Implement the cluster centroids and random undersampling techniques 
# with the credit card default data. Then estimate a logistic regression model 
# and report the classification evaluation metrics from both sampling methods.

# ln_balance_limit is the log of the maximum balance they can have on the card; 
# 1 is female, 0 male for sex; 
# the education is denoted: 1 = graduate school; 2 = university; 3 = high school; 4 = others; 
# 1 is married and 0 single for marriage; 
# default_next_month is whether the person defaults in the following month (1 yes, 0 no).
#%%
import pandas as pd
from pathlib import Path
from collections import Counter
#%%
data = Path('./Resources/cc_default.csv')
df = pd.read_csv(data)
df.head()
# %%
# drop non-predictor columns and assign features to X and target to y
x_cols = [i for i in df.columns if i not in ('ID', 'default_next_month')]
X = df[x_cols]
y = df['default_next_month']
# %%
# split the data for undersampling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train.shape
Counter(y_train)
# very unbalanced
# %%
from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
# balanced by down-selecting feature data
# %%
# Fit a Logistic regression model using random undersampled data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
# %%
# make predictions
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
# then assess results
confusion_matrix(y_test, y_pred)
# %%
# another accuracy assessment
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)
# %%
# get classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
# %%
# Fit the data using `ClusterCentroids` and check the count of each class
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
# %%
# Logistic regression using cluster centroid undersampled data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=78)
model.fit(X_resampled, y_resampled)
# %%
# generate the metrics 
#%%
# Display the confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)
#%%
# Calculate the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)
#%%
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
# %%
# start over sample
# %%
