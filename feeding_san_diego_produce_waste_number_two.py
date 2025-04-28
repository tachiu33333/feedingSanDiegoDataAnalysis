import pandas as pd
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
class DocumentData:
    def __init__(self, dataframe: pd.DataFrame, date: str):
        self.date = date
        self.dataframe = dataframe
        #self.dataframe = 

#    def __repr__(self):
#        return f"DocumentData(date={self.date}, dataframe_shape={self.dataframe.shape})"



a= pd.read_excel("2024.06 June Waste Log.xlsx", sheet_name= 2, header=6)
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
#a = a.fillna(0)

a = a[['Produce Type', 'Dropped', 'Returns']].groupby('Produce Type').agg({
    'Dropped': ['sum', 'count'],
    'Returns': ['sum', 'count']
})


a = a.reset_index().rename(columns={'index': 'Produce Type'})
a['Date'] = pd.to_datetime('2024-06-25')
a = a[['Date', 'Produce Type', 'Dropped', 'Returns']]

a.columns = ['Date', 'Produce Type', 'Dropped_sum', 'Dropped_count', 'Returns_sum', 'Returns_count']

b = pd.merge(df_shifts, a, on='Date', how='left')
print(b)

X = b[['total', 'Produce Type']]

y = b['Dropped_sum']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Produce Type']),
        ('num', SimpleImputer(strategy='constant', fill_value=0), ['total'])
    ])

model = make_pipeline(preprocessor, RandomForestRegressor())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual values:", y_test.values)