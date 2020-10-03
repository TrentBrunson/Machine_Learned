#%%
import pandas as pd
from pathlib import Path
import numpy as np

file_path = Path("../Resources/loans_data_encoded.csv")
encoded_df = pd.read_csv(file_path)
encoded_df.head()
# %%
# some models, like SVM, are sensitive to large numerical values
# SVM measures distances between data points and would trip on this

# import standardScaler to scale data in DF
from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()
# %%
# train the scaler & transform in single step
loans_data_scaled = data_scaler.fit_transform(encoded_df)
loans_data_scaled[:5]
# %%
# find mean and SD of amount
print(np.mean(loans_data_scaled[:,0])) # loans_data_scaled[:,0] returns all rows and the first column of the dataset
print(np.std(loans_data_scaled[:,0]))
# if mean approaches 0 and the SD approaches 1, the column was standardized
# %%
loans_data_scaled.shape
type(loans_data_scaled)
# %%
# create for loop to check means and SD of all columns
new_df = pd.DataFrame(columns=['mean', 'std_dev'])
new_df.head()
# %%
avg_column = loans_data_scaled.mean(axis=0)
SD_column = loans_data_scaled.std(axis=0)
print(avg_column)
print(SD_column)
#%%
x = loans_data_scaled.shape[1]
for z in range(0, x):
   m = np.mean(loans_data_scaled[:,z])
   sd = np.std(loans_data_scaled[:,z])
   print(m,sd)
#%%
for column in loans_data_scaled:
   m = np.mean(loans_data_scaled)
   print(m)
   # new_df.append(m)
   sd = np.std(loans_data_scaled)
   # new_df.append(sd)
# %%
