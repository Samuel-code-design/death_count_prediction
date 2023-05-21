import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("preprocessed_pandemic_events.csv")

# select x and y for feature selection
X = df[['duration_years', 'number_of_affected_continents', 
                  'years_ago_ended', 'disease_encoded', 'country_encoded', 'affected_continents_encoded']]

y = df['death_toll_cleaned']

# -----------------finding the best predictive features -----------

# ----Recursive Feature Elimination:
def select_features_correcaltion_RFE():
    # Create an instance of the Linear Regression model
    model = LinearRegression()

    # Create an instance of the RFE model, specifying the desired number of features to select
    rfe = RFE(model, n_features_to_select=3)

    # Fit the RFE model to your data
    rfe.fit(X, y)

    # Retrieve the selected features
    selected_features_rfe = X.columns[rfe.support_]
    print("Selected Features (RFE):", selected_features_rfe)


def select_features_correcaltion_RFR():
    # --- Random Forest Regression:
    # Create an instance of the Random Forest Regression model
    model = RandomForestRegressor()

    # Fit the model to your data
    model.fit(X, y)

    # Retrieve the feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame to store feature importances
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance score in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Retrieve the top 3 features
    selected_features_importance = importance_df['Feature'][:3].tolist()
    print("Selected Features (Feature Importances):", selected_features_importance)


def select_features_correcaltion():
    # # ---- corralations:
    # Calculate the correlation coefficients between features and the target variable
    correlation_matrix = np.corrcoef(X.values.T, y)
    correlation_with_target = abs(correlation_matrix[-1, :-1])

    # Sort the correlation coefficients in descending order
    sorted_indices = np.argsort(correlation_with_target)[::-1]

    # Retrieve the top 3 features with the highest correlation values
    selected_features_correlation = X.columns[sorted_indices[:3]].tolist()
    print("Selected Features (Correlation Analysis):", selected_features_correlation)

select_features_correcaltion()
select_features_correcaltion_RFR()
select_features_correcaltion_RFE()

# the selected features using RFE are: 'number_of_affected_continents', 'disease_encoded', 'affected_continents_encoded'
# the selected features using RFR are: 'years_ago_ended', 'duration_years', 'number_of_affected_continents'
# the selected features ing correlations are:'years_ago_ended', 'number_of_affected_continents', 'duration_years'

# the top 3 features are:
# number_of_affected_continents
# years_ago_ended
# duration_years'

# we are going to add this feature asswell for better results
# affected_continents_encoded

# --------------- pipeline voor de RFE






















