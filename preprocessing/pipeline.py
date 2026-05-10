import pandas as pd
from preprocessing.numerical import clean_numerical
from preprocessing.categorical import clean_categorical


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop(["ID", "Name", "SSN", "Annual_Income"], axis=1, inplace=True)

    df = clean_numerical(df)
    df = clean_categorical(df)

    return df
