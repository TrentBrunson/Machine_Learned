#%%
# steps of log regression (classification) model
# 1. Create a model with LogisticRegression().
# 2. Train the model with model.fit().
# 3. Make predictions with model.predict().
# 4. Validate the model with accuracy_score().
from pathlib import Path
import pandas as pd
# %%
data = Path('../Resources/diabetes.csv')
df = pd.read_csv(data)
df.head()
# %%
# Separate the Features (X) from the Target (y)
y = df["Outcome"]
X = df.drop(columns="Outcome")
# %%
# Split data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)
X_train.shape
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
# 3. make predictions
y_pred = classifier.predict(X_test)
results = pd.DataFrame({"Prediction": y_pred, "Actual": y_test}).reset_index(drop=True)
results.head(20)
# %%
# 4. validate model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))