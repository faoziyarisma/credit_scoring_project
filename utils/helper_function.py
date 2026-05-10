import re
import numpy as np
import utils.helper_variable as var
from sklearn.impute import SimpleImputer


# Cleaning the data
def return_null(val):
    if isinstance(val, str) and "__" in val:
        return np.NaN
    elif isinstance(val, str) and "_" in val:
        return np.NaN
    else:
        return val


def cal_history_age(val):
    try:
        year, month = 0, 0
        if re.search("year", val, re.IGNORECASE):
            year = int(re.findall("\d+", val)[0])
        if re.search("month", val, re.IGNORECASE):
            month = int(re.findall("\d+", val)[1])
        return year * 12 + month
    except:
        return np.NaN


def check_invalid_values(df, invalid_value, column):
    df = df.copy()
    customer_id = (
        df[(df[column] == invalid_value)]
        .groupby(by="Customer_ID", as_index=False)["Num_Bank_Accounts"]
        .mean()
    )
    print(df[df.Customer_ID.isin(customer_id.Customer_ID)])


def cleanse_from_invalid_values(new_df):
    new_df["Age"] = new_df.Age.apply(lambda x: np.NaN if x < 0 or x > 90 else x)
    new_df["Num_Bank_Accounts"] = new_df.Num_Bank_Accounts.apply(
        lambda x: np.NaN if x < 0 or x > 15 else x
    )
    new_df["Num_Credit_Card"] = new_df.Num_Credit_Card.apply(
        lambda x: np.NaN if x < 0 or x > 15 else x
    )
    new_df["Interest_Rate"] = new_df.Interest_Rate.apply(
        lambda x: np.NaN if x > 50 else x
    )
    new_df["Num_of_Loan"] = new_df.Num_of_Loan.apply(
        lambda x: np.NaN if x < 0 or x > 10 else x
    )
    new_df["Delay_from_due_date"] = new_df.Delay_from_due_date.apply(
        lambda x: 0.0 if x < 0 else x
    )
    new_df["Num_of_Delayed_Payment"] = new_df.Num_of_Delayed_Payment.apply(
        lambda x: np.NaN if x < 0 or x > 30 else x
    )
    new_df["Changed_Credit_Limit"] = new_df.Changed_Credit_Limit.apply(
        lambda x: 0.0 if x < 0 else x
    )
    new_df["Num_Credit_Inquiries"] = new_df.Num_Credit_Inquiries.apply(
        lambda x: np.NaN if x > 25 else x
    )
    new_df["Total_EMI_per_month"] = new_df.Total_EMI_per_month.apply(
        lambda x: np.NaN if x > 1400 else x
    )
    return new_df


def fill_numerical_missing_value(column, how, df):
    df = df.copy()

    missing_customer_id = df[df[column].isna()].Customer_ID
    new = (
        df[df.Customer_ID.isin(missing_customer_id)]
        .groupby(by="Customer_ID", as_index=False)
        .agg({column: how})
    )

    for index, row in df[df[column].isna()].iterrows():
        df[column].iloc[index] = new[new.Customer_ID == str(row["Customer_ID"])][column]

    return df[column]


def impute_missing_values(df):
    imputer = SimpleImputer(strategy="median")
    df[var.missing_numerical_columns] = imputer.fit_transform(
        df[var.missing_numerical_columns]
    )
    return df
