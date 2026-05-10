import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import utils.helper_function as hf
import utils.helper_variable as var
import visualizing.visualize as vz
import visualizing.eda as eda
import preprocessing.preparing as prep
import modelling.hyperparameter_tuning as ht
import modelling.modelling as mdl
import modelling.evaluating as evl

# =====================
# Load dataset
# =====================
df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
new_df = df.copy()

# =====================
# Clean numerical columns
# =====================
for col in var.invalid_numerical_columns:
    new_df[col] = new_df[col].apply(hf.return_null).astype(float)

# =====================
# Credit History Age
# =====================
new_df["Credit_History_Age"] = new_df["Credit_History_Age"].apply(hf.cal_history_age)

# =====================
# Drop unused columns
# =====================
new_df.drop(["ID", "Name", "SSN", "Annual_Income"], axis=1, inplace=True)

# =====================
# Clean invalid values
# =====================
new_df = hf.cleanse_from_invalid_values(new_df)
new_df = hf.impute_missing_values(new_df)

# =====================
# Monthly Balance (by Customer)
# =====================
new_df["Monthly_Balance"] = (
    new_df.groupby("Customer_ID")["Monthly_Balance"]
    .transform(lambda v: v.ffill().bfill())
    .astype(float)
)

# =====================
# Credit History Age (by Customer)
# =====================
new_df["Credit_History_Age"] = (
    new_df.groupby("Customer_ID")["Credit_History_Age"]
    .transform(lambda v: v.ffill().bfill())
    .astype(float)
)

# =====================
# Clean categorical columns
# =====================
new_df["Occupation"] = new_df["Occupation"].replace("_______", np.nan)
new_df["Occupation"] = new_df.groupby("Customer_ID")["Occupation"].transform(
    lambda v: v.ffill().bfill()
)

new_df["Credit_Mix"] = new_df["Credit_Mix"].str.replace("_", "Unknown", regex=False)
new_df["Payment_Behaviour"] = new_df["Payment_Behaviour"].str.replace(
    "!@9#%8", "Unknown", regex=False
)

# =====================
# Type_of_Loan (SAFE & ALIGN)
# =====================
new_df["Type_of_Loan"] = new_df["Type_of_Loan"].astype(str).str.split(",").str[0]
new_df["Type_of_Loan"] = new_df["Type_of_Loan"].replace("nan", np.nan)
new_df["Type_of_Loan"].fillna("Not Specified", inplace=True)

# =====================
# Final validation
# =====================
print(new_df[var.categorical_columns].isna().sum())

# plot the data
# vz.visualize(new_df)

# prepare data for modeling
cleaned_df = new_df.copy()
new_cleaned_df = cleaned_df.drop(
    columns=[
        "Customer_ID",
        "Month",
        "Occupation",
        "Type_of_Loan",
        "Credit_Utilization_Ratio",
    ],
    axis=1,
)
new_cleaned_df.head()

# preprocessing for modeling
train_df, test_df = train_test_split(
    new_cleaned_df, test_size=0.05, random_state=42, shuffle=True
)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print(train_df.shape)
print(test_df.shape)

# eda on train_df
# eda.eda(train_df)

# prepare data for modeling
undersampled_train_df = prep.cluster_data(train_df)
X_train, X_test, y_train, y_test = prep.splitting_data(undersampled_train_df, test_df)
new_train_df, new_test_df = prep.transform_fields_data(X_train, X_test)
new_y_train, new_y_test = prep.transform_target_data(y_train, y_test)

# Apply PCA for dimensionality reduction
train_pca_df = new_train_df.copy().reset_index(drop=True)
test_pca_df = new_test_df.copy().reset_index(drop=True)

# Determine the best number of PCA components for numerical columns 1
prep.det_best_pca_comp(var.pca_numerical_columns_1, train_pca_df)
# from the result, we can see that 5 components can explain 90% of the variance, so we will use 5 components for PCA
n_components_num_col_1 = 5
train_pca_df = prep.apply_pca(
    var.pca_numerical_columns_1, train_pca_df, n_components=n_components_num_col_1
)

# Determine the best number of PCA components for numerical columns 2
prep.det_best_pca_comp(var.pca_numerical_columns_2, train_pca_df)
# from the result, we can see that 5 components can explain 90% of the variance, so we will use 5 components for PCA
n_components_num_col_2 = 2
train_pca_df = prep.apply_pca(
    var.pca_numerical_columns_2, train_pca_df, n_components=n_components_num_col_2
)

# Determine Optimal Hyperparameters
# Decision Tree
# best_params = ht.decision_tree_gridSearch(train_pca_df, new_y_train)

# # Random Forest
# best_params = ht.random_forest_gridSearch(train_pca_df, new_y_train)

# # Gradient Boosting
# best_params = ht.gradient_boosting_gridSearch(train_pca_df, new_y_train)

# Train the model with optimal hyperparameters
# Decision Tree
tree_model = mdl.decision_tree_with_optimal_hyperparam(train_pca_df, new_y_train)
# evl.evaluating_all(tree_model, X_test, y_test)

# Random Forest
# rdf_model = mdl.random_forest_with_optimal_hyperparam(train_pca_df, new_y_train)

# Gradient Boosting
gboost_model = mdl.gradient_boosting_with_optimal_hyperparam(train_pca_df, new_y_train)
evl.plot_feature_importances(gboost_model.feature_importances_, train_pca_df.columns)
