#%%
# Initial imports
import pandas as pd
from pathlib import Path
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# %%
# Loading data
file_path = Path("../Resources/loans_data_encoded.csv")
df_loans = pd.read_csv(file_path)
df_loans.head()
# %%
# Define the features set.
X = df_loans.copy()
X = X.drop("bad", axis=1)
X.head()
# %%
# Define the target set.
y = df_loans["bad"].values
y[:5]
# %%
# Splitting into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=78)
# %%
# Determine the shape of our training and testing sets.
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
# Splitting into Train and Test sets into an 80/20 split.
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, random_state=78, train_size=0.80)
# %%
# Determine the shape of our training and testing sets.
print(X_train2.shape)
print(X_test2.shape)
print(y_train2.shape)
print(y_test2.shape)
# %%
# scaling the data so if want to compare this decision tree
# model to other best fit models, can do so quicky

# Creating a StandardScaler instance.
scaler = StandardScaler()

# Fitting the Standard Scaler with the training data.
X_scaler = scaler.fit(X_train)

# Scaling the data.
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# %%
# verify that the mean of each column is 0 
# and its standard deviation is 1
import numpy as np
print(np.mean(X_train_scaled[:,0]))
print(np.mean(X_test_scaled[:,0]))
print(np.std(X_train_scaled[:,0]))
print(np.std(X_test_scaled[:,0]))
# %%
# Fitting the Decision Tree Model

# Random Forest
# Create a random forest classifier.
rf_model = RandomForestClassifier(n_estimators=500, random_state=78) 

# Fitting the random forest model
rf_model = rf_model.fit(X_train_scaled, y_train)
# %%
# Making Predictions Using the Tree Model

# Making predictions using the random forest testing data.
predictions = rf_model.predict(X_test_scaled)
#  %%
# Model Evaluation

# %%
# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual Good (0)", "Actual Bad (1)"], columns=["Predicted Good (0)", "Predicted Bad(1)"])

cm_df
# %%
    # 	            Predicted True	Predicted False
# Actually True	    TRUE POSITIVE	FALSE NEGATIVE
# Actually False    FALSE POSITIVE	TRUE NEGATIVE

# Precision = TP/(TP + FP)
# Sensitivity (recall) = TP/(TP + FN)

# F1 score (harmonic mean) = 2(Precision * Sensitivity)/(Precision + Sensitivity)

# %%
# add the columns and the rows
total_column = cm_df.sum(axis = 1)
total_row = cm_df.sum(axis = 0)

total_column
total_row

# add the new totals to the DF
cm_df['Column Totals'] = total_column
cm_df.loc['Row Totals'] = cm_df.sum()

# cm_df = cm_df.drop(columns='Column Totals')
cm_df
# %%
# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)
acc_score
# %%
# Displaying results
print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))
# %%
# rank order the features in the random forest
importances = rf_model.feature_importances_
importances
# %%
# sort the features by their importance.
sorted(zip(rf_model.feature_importances_, X.columns), 
reverse=True)
# %%
