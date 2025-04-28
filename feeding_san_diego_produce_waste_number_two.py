import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

#volunteer data
df = pd.read_csv('new_volunteer_data.csv', low_memory=False)
df = df[["Volunteer ID", "Date", "Time", "Current Status", "Opportunity Tags"]]

df = df[df["Current Status"] == "Completed"]
df = df[df["Opportunity Tags"].str.contains("Food Sorting", case = False, na=False)]
df.drop(["Opportunity Tags","Current Status"], axis = 1, inplace = True)


print(df)

column_trial = df.loc[:,"Time"]

column_trial = pd.to_datetime(column_trial).dt.hour

def categorize_time(t):
    if t < 13:
        return 'morning'
    elif t < 16:
        return 'afternoon'
    else:
        return 'evening'

df['Shift'] = column_trial.apply(categorize_time)

df_shifts = df.pivot_table(index='Date', columns='Shift', values='Volunteer ID', aggfunc='count')

#print(df_shifts)

#waste log data
class DocumentData:
    def __init__(self, dataframe: pd.DataFrame, date: str):
        self.date = date
        self.dataframe = dataframe
        #self.dataframe = 

#    def __repr__(self):
#        return f"DocumentData(date={self.date}, dataframe_shape={self.dataframe.shape})"



a= pd.read_excel("2024.06 June Waste Log.xlsx", sheet_name= 0, header=6)
a = a.iloc[:, :-1]

resource = pd.read_excel("2024.06 June Waste Log.xlsx", sheet_name= 1, header = None)
resource = resource[0].to_list()


first_col = a.columns[0]
a[first_col] = a[first_col].replace('enter >>', np.nan)

current_resource = None
resource_col = []

for val in a[first_col]:
    if val in resource:
        current_resource = val
        resource_col.append(pd.NA)
    else:
        resource_col.append(current_resource)
a[first_col] = resource_col
a = a.dropna(subset=[first_col])


print(a[['Produce Type', 'Dropped', 'Returns']])
