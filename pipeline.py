import pandas as pd
import datetime
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import time

start_time = time.time()

# load cleaned dataframe from kaggle:
df = pd.read_csv("cleaned_pandemic_events.csv") #cleaned Chronological Table of Epidemic and Pandemic Events in human History

# clean column names:
df.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
                                # columns:
                                # 'unnamed:_0', 'event', 'date', 'location', 'disease',
                                #        'death_toll_(estimate)', 'is_death_toll_unknown',
                                #        'death_toll_lower_limit', 'death_toll_upper_limit', 'bc/ad',
                                #        'latestdate', 'earliestdate', 'disease_cleaned', 'country',
                                #        'europe_affected', 'asia_affected', 'africa_affected',
                                #        'oceania_affected', 'north_america_affected', 'south_america_affected'

# # remove rows with missing data that can not be filled in
df = df.drop(df[df['is_death_toll_unknown'] == True].index)
df = df.drop(df[df['disease_cleaned'] == 'Unknown'].index)

# create catigorical columns 
df["disease_cleaned"]=pd.Categorical(df["disease_cleaned"])
df["country"]=pd.Categorical(df["country"])

# add duration columm
df['duration_years'] = df['latestdate'] - df['earliestdate']
# use bc and ac, if bc then earliest - latest

# Adjust the duration for BC dates
bc_mask = df['bc/ad'] == 'BC'
df.loc[bc_mask, 'duration_years'] *= -1

 # use death toll average of lower limit and upper limit to create a cleaned death toll column
df['death_toll_cleaned'] = (df['death_toll_lower_limit'].fillna(df['death_toll_upper_limit']) +
                           df['death_toll_upper_limit'].fillna(df['death_toll_lower_limit'])) / 2

# Create a new catigorical column 'Affected_Continents' based on the boolean columns with affected continents
Affected_Continents = ['europe_affected', 'asia_affected', 'africa_affected', 'oceania_affected', 'north_america_affected', 'south_america_affected']

df['affected_continents'] = df[Affected_Continents].apply(lambda row: ', '.join(row[row].index.str.split('_affected').str[0]), axis=1)
df["affected_continents"]=pd.Categorical(df["affected_continents"])

# create new column number_of_affected_continents using affected_continents
df['number_of_affected_continents'] = df['affected_continents'].apply(lambda x: len(x.split(', ')))

current_year = datetime.datetime.now().year

def calculate_years_ago(row):
    if row['bc/ad'] == 'BC':
        latest_year = -row['latestdate']
        earliest_year = -row['earliestdate']
    else:
        latest_year = row['latestdate']
        earliest_year = row['earliestdate']
    return current_year - max(latest_year, earliest_year)

df['years_ago_ended'] = df.apply(calculate_years_ago, axis=1)

# remove unused columns
df.drop(['earliestdate', 'latestdate', 'bc/ad', 'europe_affected', 
         'asia_affected', 'africa_affected', 'oceania_affected', 
         'north_america_affected', 'south_america_affected', 
         'death_toll_upper_limit', 'death_toll_lower_limit', 
         'death_toll_(estimate)', 'is_death_toll_unknown', 
         'unnamed:_0', 'disease', 'date', 'location'], axis=1, inplace=True)


# ---------------encoding catigorical columns:-------------
label_encoder = LabelEncoder()
df['disease_encoded'] = label_encoder.fit_transform(df['disease_cleaned'])
df['country_encoded'] = label_encoder.fit_transform(df['country'])
df['affected_continents_encoded'] = label_encoder.fit_transform(df['affected_continents'])

# ------------selected the features in select_features.py
# df.to_csv('preprocessed_pandemic_events.csv', index=False)

# --------------------------------predicting the death toll
# Split the data into X and y with selected features
X = df[['duration_years', 'number_of_affected_continents', 'years_ago_ended', 'affected_continents_encoded']]
y = df['death_toll_cleaned']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Define the column transformer for standard scaling
standard_scaler_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['duration_years', 'number_of_affected_continents', 'years_ago_ended'])
    ])

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', standard_scaler_transformer),
                           ('regressor', LinearRegression())])

# Fit the pipeline to the training data and make predictions on the testing data
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# get the model summary with statsmodels
X_train_constant = sm.add_constant(X_train)
sm_model = sm.OLS(y_train, X_train_constant).fit()

# Print the summary, metrics, look at adjusted R-squared since it is a multivariate regression
print(sm_model.summary()) 
print(f"mse: {mse} mae: {mae}")

end_time = time.time()

execution_time = end_time - start_time
print("Pipeline execution time:", execution_time, "seconds")