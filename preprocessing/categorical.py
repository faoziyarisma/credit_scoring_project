import numpy as np
import pandas as pd


def clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Monthly Balance
    df["Monthly_Balance"] = (
        df.groupby("Customer_ID")["Monthly_Balance"]
        .transform(lambda v: v.ffill().bfill())
        .astype(float)
    )

    # Credit History Age
    df["Credit_History_Age"] = (
        df.groupby("Customer_ID")["Credit_History_Age"]
        .transform(lambda v: v.ffill().bfill())
        .astype(float)
    )

    # Occupation
    df["Occupation"] = df["Occupation"].replace("_______", np.nan)
    df["Occupation"] = df.groupby("Customer_ID")["Occupation"].transform(
        lambda v: v.ffill().bfill()
    )

    # Credit Mix & Payment Behaviour
    df["Credit_Mix"] = df["Credit_Mix"].str.replace("_", "Unknown", regex=False)
    df["Payment_Behaviour"] = df["Payment_Behaviour"].str.replace(
        "!@9#%8", "Unknown", regex=False
    )

    # Type of Loan
    df["Type_of_Loan"] = df["Type_of_Loan"].astype(str).str.split(",").str[0]
    df["Type_of_Loan"] = df["Type_of_Loan"].replace("nan", np.nan)
    df["Type_of_Loan"].fillna("Not Specified", inplace=True)

    return df
