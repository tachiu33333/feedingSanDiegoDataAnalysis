import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


#format the volunteers into something simple
def create_total_df(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df = df[["Volunteer ID", "Date", "Time", "Current Status", "Opportunity Tags"]]

    df = df[df["Current Status"] == "Completed"]
    df = df[df["Opportunity Tags"].str.contains("Food Sorting", case = False, na=False)]
    df.drop(["Opportunity Tags","Current Status"], axis = 1, inplace = True)

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
    df_shifts['total'] = df_shifts['afternoon'].fillna(0) + df_shifts['evening'].fillna(0) + df_shifts['morning'].fillna(0)
    df_shifts = df_shifts.reset_index().rename(columns={'index': 'Date'})
    df_shifts['Date'] = pd.to_datetime(df_shifts['Date'], format='%m/%d/%Y')

    return df_shifts

df_shifts = create_total_df('new_volunteer_data.csv')

#waste log data
resource = pd.read_excel("2024.06 June Waste Log.xlsx", sheet_name= 1, header = None)
resource = resource[0].to_list()

def fix_waste_log(filepath, date, page):
    waste = pd.read_excel(filepath, sheet_name= page, header=6).iloc[:, :-1]

    first_col = waste.columns[0]
    waste[first_col] = waste[first_col].replace('enter >>', np.nan)

    current_resource = None
    resource_col = []

    for val in waste[first_col]:
        if val in resource:
            current_resource = val
            resource_col.append(pd.NA)
        else:
            resource_col.append(current_resource)
    waste[first_col] = resource_col
    waste = waste.dropna(subset=[first_col])
    waste = waste.iloc[:, :3]
    waste['Returns'] = pd.to_numeric(waste['Returns'], errors='coerce')
    waste['Dropped'] = pd.to_numeric(waste['Dropped'], errors='coerce')
    waste = waste[['Produce Type', 'Dropped', 'Returns']].groupby('Produce Type').agg({
    'Dropped': ['sum', 'count'],
    'Returns': ['sum', 'count']
    })


    waste = waste.reset_index().rename(columns={'index': 'Produce Type'})
    waste['Date'] = pd.to_datetime(date)
    waste = waste[['Date', 'Produce Type', 'Dropped', 'Returns']]

    waste.columns = ['Date', 'Produce Type', 'Dropped_sum', 'Dropped_count', 'Returns_sum', 'Returns_count']
    return waste

all_data = []
for date, sheet in [('2024-06-25', 2), ('2024-06-26', 3), ('2024-06-27', 4), ('2024-07-02', 5)]:
    all_data.append(fix_waste_log("2024.06 June Waste Log.xlsx", date, sheet))

for date, sheet in [('2024-07-03', 0), ('2024-07-05', 1), ('2024-07-06', 2), ('2024-07-08', 3),
                    ('2024-07-09', 4), ('2024-07-10', 5), ('2024-07-11', 6), ('2024-07-12', 7),
                    ('2024-07-13', 8), ('2024-07-15', 9), ('2024-07-17', 10), ('2024-07-18', 11),
                    ('2024-07-19', 12), ('2024-07-20', 13), ('2024-07-22', 14), ('2024-07-23', 15),
                    ('2024-07-24', 16), ('2024-07-25', 17), ('2024-07-26', 18), ('2024-07-27', 19),
                    ('2024-07-29', 20), ('2024-07-30', 21), ('2024-07-31', 22)]:
    all_data.append(fix_waste_log("2024.07 July Produce Waste Log.xlsx", date, sheet))
combined = pd.concat(all_data, ignore_index=True)

b = pd.merge(df_shifts, combined, on='Date', how='inner')
print(b)

#X = b[['total', 'Produce Type']]

#y = b['Dropped_sum']

#preprocessor = ColumnTransformer(
#    transformers=[
#        ('cat', OneHotEncoder(), ['Produce Type']),
#        ('num', SimpleImputer(strategy='constant', fill_value=0), ['total'])
#    ])

#model = make_pipeline(preprocessor, RandomForestRegressor())

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model.fit(X_train, y_train)

#y_pred = model.predict(X_test)

#print("Predictions:", y_pred)
#print("Actual values:", y_test.values)

print(resource)

