import matplotlib.pyplot as plt
import seaborn as sns
import utils.helper_variable as var


def categorical_plot(features, df, segment_feature=None):
    fig, ax = plt.subplots(len(features), 1, figsize=(10, 20))
    for i, feature in enumerate(features):
        if segment_feature:
            sns.countplot(data=df, y=segment_feature, hue=feature, ax=ax[i])
        else:
            sns.countplot(data=df, x=feature, ax=ax[i])
    plt.tight_layout()
    plt.show()


def numerical_dis_plot(features, df, segment_feature=None, showfliers=True):
    fig, ax = plt.subplots(len(features), 1, figsize=(15, 30))
    for i, feature in enumerate(features):
        if segment_feature:
            sns.boxplot(
                y=segment_feature, x=feature, data=df, ax=ax[i], showfliers=showfliers
            )
            ax[i].set_ylabel(None)
        else:
            sns.boxplot(x=feature, data=df, ax=ax[i], showfliers=showfliers)
    plt.tight_layout()
    plt.show()


def visualize(new_df):
    fig, ax = plt.subplots(len(var.categorical_columns), 1, figsize=(10, 24))
    for i, feature in enumerate(var.categorical_columns):
        sns.countplot(data=new_df, y=feature, ax=ax[i])
    plt.show()

    numerical_dis_plot(features=var.numerical_columns, df=new_df)

    numerical_dis_plot(
        features=var.numerical_columns, df=new_df, segment_feature="Credit_Score"
    )

    categorical_plot(
        features=var.categorical_columns,
        df=new_df,
        segment_feature="Credit_Score",
    )

    plt.figure(figsize=(20, 20))
    sns.heatmap(
        new_df[var.numerical_columns].corr(),
        annot=True,
        cmap="jet",
        linecolor="black",
        linewidth=1,
    )
    plt.show()
    return None
