#%%
import pandas as pd
from pathlib import Path

file_path = Path("../Resources/loans_data.csv")
loans_df = pd.read_csv(file_path)
loans_df.head()
# %%
# Dataset Information
# The file loans_data.csv, contains simulated data about loans, there are a total of 500 records. Each row represents a loan application along an arbitrary year, where every column represents the following data about every loan application.

# amount: The loan amount in USD.
# term: The loan term in months.
# month: The month of the year when the loan was requested.
# age: Age of the loan applicant.
# education: Educational level of the loan applicant.
# gender: Gender of the loan applicant.
# bad: Stands for a bad or good loan applicant (1 - bad, 0 - good).
# %%
# begin encoding

# %%
# Binary encoding using Pandas (single column)
loans_binary_encoded = pd.get_dummies(loans_df, columns=["gender"])
loans_binary_encoded.head()
# %%
# encode multiple columns with binary results at same time
loans_binary_encoded = pd.get_dummies(loans_df, columns=["education", "gender"])
loans_binary_encoded.head()
# %%
# do same with Scikit-learn's built in label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2 = loans_df.copy()
df2['education'] = le.fit_transform(df2['education']) 
df2.head()
# changed education 0=bachelor, 1=high school or below, 
# 2 = college, 3 = master or above
# %%
# apply to gender
df2['gender'] = le.fit_transform(df2['gender'])
df2.head()
# changed gender to 1 or 0; female = 0, male = 1
# %%
# custom label encoding with scikit LabelEncoder

# old way
loans_df['month_le'] = le.fit_transform(loans_df['month'])
loans_df.head()
# assigned values to new month_le column based on 
# alphabetical order of previous month column
# %%
# want months in numerical order, create dictionary of key-value pairs
months_num = {
   "January": 1,
   "February": 2,
   "March": 3,
   "April": 4,
   "May": 5,
   "June": 6,
   "July": 7,
   "August": 8,
   "September": 9,
   "October": 10,
   "November": 11,
   "December": 12,
}
# %%
# use lamda (anonymous) function on month column to perform conversion to numeric values
loans_df["month_num"] = loans_df["month"].apply(lambda x: months_num[x])
loans_df.head()
# %%
# drop the old columns now that it's converted correctly
loans_df = loans_df.drop(["month", "month_le"], axis=1)
loans_df.head()
# %%
