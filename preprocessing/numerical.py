import pandas as pd
import utils.helper_function as hf
import utils.helper_variable as var


def clean_numerical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in var.invalid_numerical_columns:
        df[col] = df[col].apply(hf.return_null).astype(float)

    df["Credit_History_Age"] = df["Credit_History_Age"].apply(hf.cal_history_age)

    df = hf.cleanse_from_invalid_values(df)
    df = hf.impute_missing_values(df)

    return df
