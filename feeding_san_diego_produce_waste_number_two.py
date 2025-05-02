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
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import pickle


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

#format waste log data
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

def set_up_waste_logs():
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

    for date, sheet in [('2024-08-01', 0), ('2024-08-02', 1), ('2024-08-03', 2), ('2024-08-05', 3),
                        ('2024-08-06', 4), ('2024-08-07', 5), ('2024-08-08', 6), ('2024-08-09', 7),
                        ('2024-08-10', 8), ('2024-08-12', 9), ('2024-08-13', 10),('2024-08-14', 11),
                        ('2024-08-15', 12),('2024-08-16', 13),('2024-08-17', 14),('2024-08-19', 15),
                        ('2024-08-20', 16),('2024-08-21', 17),('2024-08-22', 18),('2024-08-23', 19),
                        ('2024-08-24', 20),('2024-08-26', 21),('2024-08-27', 22),('2024-08-28', 23),
                        ('2024-08-29', 24),('2024-08-30', 25),('2024-08-31', 26)]:
        all_data.append(fix_waste_log("2024.08 August Produce Waste Log.xlsx", date, sheet))

#    for date, sheet in [('2024-09-03', 0), ('2024-09-04', 1), ('2024-09-05', 2), ('2024-09-06', 3),
#                        ('2024-09-07', 4), ('2024-09-09', 5), ('2024-09-10', 6), ('2024-09-11', 7),
#                        ('2024-09-12', 8), ('2024-09-13', 9), ('2024-09-14', 10),('2024-09-16', 11),
#                        ('2024-09-17', 12),('2024-09-18', 13),('2024-09-19', 14),('2024-09-20', 15),
#                        ('2024-09-20', 16),('2024-09-21', 17),('2024-09-22', 18),('2024-09-23', 19),
#                        ('2024-09-23', 20),('2024-09-24', 21),('2024-09-26', 22),('2024-09-27', 23),
#                        ('2024-09-28', 24)]:
#        all_data.append(fix_waste_log("2024.09 September Produce Waste Log.xlsx", date, sheet))

    for date, sheet in [('2024-09-30', 0), ('2024-10-01', 1), ('2024-10-02', 2), ('2024-10-03', 3),
                        ('2024-10-04', 4), ('2024-10-05', 5), ('2024-10-07', 6), ('2024-10-08', 7),
                        ('2024-10-09', 8), ('2024-10-10', 9), ('2024-10-11', 10),('2024-10-12', 11),
                        ('2024-10-14', 12),('2024-10-15', 13),('2024-10-16', 14),('2024-10-17', 15),
                        ('2024-10-18', 16),('2024-10-19', 17),('2024-10-21', 18),('2024-10-22', 19),
                        ('2024-10-24', 20),('2024-10-25', 21),('2024-10-26', 22),('2024-10-28', 23),
                        ('2024-10-29', 24),('2024-10-30', 25),('2024-10-31', 26), ('2024-11-01', 27),('2024-11-02', 28)]:
        all_data.append(fix_waste_log("2024.10 October Produce Waste Log.xlsx", date, sheet))

    return pd.concat(all_data, ignore_index=True)

full_set = pd.merge(df_shifts, set_up_waste_logs(), on='Date', how='inner')
print(full_set)

X = full_set[['total', 'Produce Type']]  # Features (total and Produce Type)
y = full_set[['Dropped_sum', 'Returns_sum']]  # Target variables (Dropped_sum and Returns_sum)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocessing - one-hot encoding for categorical variable 'Produce Type'
categorical = ['Produce Type']
numerical = ['total']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numerical)
    ]
)

# Step 4: Create the model pipeline with MultiOutputRegressor
model = make_pipeline(
    preprocessor, 
    MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model performance for both targets
#mse_dropped = mean_squared_error(y_test['Dropped_sum'], y_pred[:, 0])
#mse_returns = mean_squared_error(y_test['Returns_sum'], y_pred[:, 1])

#print(f'Mean Squared Error for Dropped_sum: {mse_dropped}')
#print(f'Mean Squared Error for Returns_sum: {mse_returns}')

with open("drop_predictor.pkl", "wb") as f:
    pickle.dump(model, f)
