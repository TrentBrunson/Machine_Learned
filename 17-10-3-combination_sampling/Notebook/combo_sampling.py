#%%
# SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. 
# SMOTEENN is a two-step process:

# Oversample the minority class with SMOTE.
    # Clean the resulting data with an undersampling strategy. 
    # If the two nearest neighbors of a data point belong to two different classes, 
    # that data point is dropped.
# %%
import pandas as pd
from path import Path
from collections import Counter
#%%
# import the data
data = Path('../Resources/cc_default.csv')
df = pd.read_csv(data)
df.head()
# %%
# assign the data into features and target
x_cols = [i for i in df.columns if i not in ('ID', 'default_next_month')]
X = df[x_cols]
y = df['default_next_month']
#%%
x_cols
# %%
Counter(y)
# %%
# Normal train-test split
# split the data into training & target sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# %%
# Combination sampling with SMOTEENN
# Use the SMOTEENN technique to perform combination sampling on the data
# Count the resampled classes
# using TensorFlow back end
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
Counter(y_resampled)
# %%
# Logistic Regression
# Fit a Logistic regression model using random undersampled data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
# %%
# Evaluate metrics
# Display the confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)
# %%
# Calculate the Balanced Accuracy Score
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)
# %%
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
# %%
# similar results to over sampling which did better than under sampling
