import matplotlib.pyplot as plt
import seaborn as sns


def eda(train_df):
    sns.countplot(data=train_df, x="Credit_Score")
    plt.show()

    print(train_df.Credit_Score.value_counts())
    return None
