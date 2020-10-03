#%%
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

file_path = Path("../Resources/loans_data.csv")
loans_df = pd.read_csv(file_path)
df2 = loans_df.copy()
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
# create dictionary for custom encoding of months
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
le = LabelEncoder()
# transform education to numerical values
loans_df['education'] = le.fit_transform(loans_df['education']) 
# transform gender to numerical values
loans_df['gender'] = le.fit_transform(loans_df['gender']) 
# use lamda (anonymous) function on month column to perform conversion to numeric values
loans_df["month_num"] = loans_df["month"].apply(lambda x: months_num[x])
loans_df.head()
#%%
# save output to csv
loans_df.to_csv('../Resources/loans_data_encoded.csv')
# %%
# do it the book way with Pandas

# create dictionary for custom encoding of months
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
# trandform gender and education
loans_binary_encoded = pd.get_dummies(df2, columns=["education", "gender"])
loans_binary_encoded
#%%

# transform months column and drop old month column
loans_binary_encoded["month_num"] = loans_binary_encoded["month"].apply(lambda x: months_num[x])
loans_binary_encoded = loans_binary_encoded.drop(["month"], axis = 1)
loans_binary_encoded.head()
# %%
list(loans_binary_encoded)
# %%
loans_binary_encoded = loans_binary_encoded[[
    'amount', 
    'term', 
    'age', 
    'bad',
    'month_num',
    'education_Bachelor',
    'education_High School or Below',
    'education_Master or Above',
    'education_college',
    'gender_female',
    'gender_male'
    ]]
loans_binary_encoded.head()
# %%
loans_binary_encoded.to_csv('../Resources/loans_data_encoded.csv')
