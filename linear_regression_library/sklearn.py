#%%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# %%
df = pd.read_csv(Path('./Resources/Salary_Data.csv'))
df.head()
# %%
