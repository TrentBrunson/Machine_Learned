#%%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
# %%
df = pd.read_csv(Path('./Resources/Salary_Data.csv'))
df.head()
# %%
plt.scatter(df.YearsExperience, df.Salary)
plt.xlabel('Years of Experience')
plt.ylabel('Salary in USD')
plt.show()
# %%
# formats the data to meet the requirements of the Scikit-learn library:
X = df.YearsExperience.values.reshape(-1, 1)
# %%
X[:5]
# %%
X.shape
# %%
y = df.Salary
# %%
model = LinearRegression()
# %%
model.fit(X, y)
# %%
y_pred = model.predict(X)
print(y_pred.shape)
# %%
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.show()
# %%
