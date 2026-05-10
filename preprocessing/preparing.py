import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
import utils.helper_variable as var


def splitting_data(undersampled_train_df, test_df):
    X_train = undersampled_train_df.drop(columns="Credit_Score", axis=1)
    y_train = undersampled_train_df["Credit_Score"]

    X_test = test_df.drop(columns="Credit_Score", axis=1)
    y_test = test_df["Credit_Score"]
    return X_train, X_test, y_train, y_test


def scaling(features, df, df_test=None):
    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()
        for feature in features:
            scaler = MinMaxScaler()
            X = np.asanyarray(df[feature])
            X = X.reshape(-1, 1)
            scaler.fit(X)
            df["{}".format(feature)] = scaler.transform(X)
            joblib.dump(scaler, "model/scaler_{}.joblib".format(feature))

            X_test = np.asanyarray(df_test[feature])
            X_test = X_test.reshape(-1, 1)
            df_test["{}".format(feature)] = scaler.transform(X_test)
        return df, df_test
    else:
        df = df.copy()
        for feature in features:
            scaler = MinMaxScaler()
            X = np.asanyarray(df[feature])
            X = X.reshape(-1, 1)
            scaler.fit(X)
            df["{}".format(feature)] = scaler.transform(X)
            joblib.dump(scaler, "model/scaler_{}.joblib".format(feature))
        return df


def encoding(features, df, df_test=None):
    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()
        for feature in features:
            encoder = LabelEncoder()
            encoder.fit(df[feature])
            df["{}".format(feature)] = encoder.transform(df[feature])
            joblib.dump(encoder, "model/encoder_{}.joblib".format(feature))

            df_test["{}".format(feature)] = encoder.transform(df_test[feature])
        return df, df_test
    else:
        df = df.copy()
        for feature in features:
            encoder = LabelEncoder()
            encoder.fit(df[feature])
            df["{}".format(feature)] = encoder.transform(df[feature])
            joblib.dump(encoder, "model/encoder_{}.joblib".format(feature))
        return df


def cluster_data(train_df):
    df_majority_1 = train_df[(train_df.Credit_Score == "Standard")]
    df_majority_2 = train_df[(train_df.Credit_Score == "Poor")]
    df_minority = train_df[(train_df.Credit_Score == "Good")]

    df_majority_1_undersampled = resample(
        df_majority_1, n_samples=16936, random_state=42
    )
    df_majority_2_undersampled = resample(
        df_majority_2, n_samples=16936, random_state=42
    )
    # print(df_majority_1_undersampled.shape)
    # print(df_majority_2_undersampled.shape)
    # print(df_minority.shape)

    undersampled_train_df = pd.concat(
        [df_minority, df_majority_1_undersampled]
    ).reset_index(drop=True)
    undersampled_train_df = pd.concat(
        [undersampled_train_df, df_majority_2_undersampled]
    ).reset_index(drop=True)
    undersampled_train_df = shuffle(undersampled_train_df, random_state=42)
    undersampled_train_df.reset_index(drop=True, inplace=True)
    print(undersampled_train_df.sample(5))

    # sns.countplot(data=undersampled_train_df, x="Credit_Score")
    # plt.show()

    return undersampled_train_df


def transform_fields_data(X_train, X_test):
    new_train_df, new_test_df = scaling(var.numerical_utilize_columns, X_train, X_test)
    new_train_df, new_test_df = encoding(
        var.categorical_utilize_columns, new_train_df, new_test_df
    )

    return new_train_df, new_test_df


def transform_target_data(y_train, y_test):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    new_y_train = encoder.transform(y_train)
    joblib.dump(encoder, "model/encoder_target.joblib")

    new_y_test = encoder.transform(y_test)
    return new_y_train, new_y_test


def det_best_pca_comp(pca_numerical_columns, pca_df):
    pca = PCA(n_components=len(pca_numerical_columns), random_state=123)
    pca.fit(pca_df[pca_numerical_columns])
    princ_comp = pca.transform(pca_df[pca_numerical_columns])

    var_exp = pca.explained_variance_ratio_.round(3)
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(
        range(len(pca_numerical_columns)),
        var_exp,
        alpha=0.5,
        align="center",
        label="individual explained variance",
    )
    plt.step(
        range(len(pca_numerical_columns)),
        cum_var_exp,
        where="mid",
        label="cumulative explained variance",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    plt.show()

    return princ_comp


def apply_pca(pca_numerical_columns, pca_df, n_components):
    # Initialize PCA
    pca_1 = PCA(n_components=n_components, random_state=123)

    # Fit PCA
    pca_1.fit(pca_df[pca_numerical_columns])

    # Save model
    joblib.dump(pca_1, f"model/pca_{n_components}.joblib")

    # Transform data
    princ_comp_1 = pca_1.transform(pca_df[pca_numerical_columns])

    # Generate dynamic column names
    pca_columns = [f"pc1_{i+1}" for i in range(n_components)]

    # Add PCA columns
    pca_df[pca_columns] = pd.DataFrame(
        princ_comp_1, columns=pca_columns, index=pca_df.index
    )

    # Drop original columns
    pca_df.drop(columns=pca_numerical_columns, inplace=True)

    print(pca_df.head())

    return pca_df
