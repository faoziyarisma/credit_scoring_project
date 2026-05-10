from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib


def evaluating(y_pred, y_true):
    """Evaluasi model"""
    labels = ["Good", "Poor", "Standard"]

    print(classification_report(y_pred=y_pred, y_true=y_true))

    cnf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)
    confusion_matrix_df = pd.DataFrame(cnf_matrix, labels, labels)
    sns.heatmap(
        confusion_matrix_df, annot=True, annot_kws={"size": 14}, fmt="d", cmap="YlGnBu"
    )
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()

    return confusion_matrix_df


def evaluating_all(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    encoder = joblib.load("model/encode_target.joblib")
    y_pred_test = encoder.inverse_transform(y_pred_test)
    evaluating(y_pred=y_pred_test, y_true=y_test)
    return None


def plot_feature_importances(feature_importances, cols):
    features = pd.DataFrame(feature_importances, columns=["coef_value"]).set_index(cols)
    features = features.sort_values(by="coef_value", ascending=False)
    top_features = features

    plt.figure(figsize=(10, 6))
    sns.barplot(x="coef_value", y=features.index, data=features)
    plt.show()
    return top_features
