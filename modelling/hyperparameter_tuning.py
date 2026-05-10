import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# =====================
# Logistic Regression
# =====================
def logistic_regression_gridSearch(train_pca_df, new_y_train):
    param_grid = {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1]}

    log_model = LogisticRegression(random_state=123)

    CV_lr = GridSearchCV(estimator=log_model, param_grid=param_grid, cv=5, n_jobs=-1)
    CV_lr.fit(train_pca_df, new_y_train)
    print("Best parameters:", CV_lr.best_params_)
    print("Best score:", CV_lr.best_score_)
    return CV_lr.best_params_


# =====================
# Decision Tree
# =====================
def decision_tree_gridSearch(train_pca_df, new_y_train):
    tree_model = DecisionTreeClassifier(random_state=123)

    param_grid = {
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [5, 6, 7, 8],
        "criterion": ["gini", "entropy"],
    }

    CV_tree = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, n_jobs=-1)
    CV_tree.fit(train_pca_df, new_y_train)
    print("Best parameters:", CV_tree.best_params_)
    print("Best score:", CV_tree.best_score_)
    return CV_tree.best_params_


# =====================
# Random Forest
# =====================
def random_forest_gridSearch(train_pca_df, new_y_train):
    rdf_model = RandomForestClassifier(random_state=123)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["sqrt", "log2"],
        "max_depth": [6, 7, 8],
        "criterion": ["gini", "entropy"],
        # "verbose": 2,
    }

    CV_rdf = GridSearchCV(estimator=rdf_model, param_grid=param_grid, cv=5, n_jobs=-1)

    CV_rdf.fit(train_pca_df, new_y_train)

    print("Best parameters:", CV_rdf.best_params_)
    print("Best score:", CV_rdf.best_score_)

    return CV_rdf.best_params_


# =====================
# Logistic Regression
# =====================
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def gradient_boosting_gridSearch(train_pca_df, new_y_train):
    gboost_model = GradientBoostingClassifier(random_state=123)

    param_grid = {
        "max_depth": [5, 8],
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_features": ["sqrt", "log2"],
    }

    CV_gboost = GridSearchCV(
        estimator=gboost_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )

    CV_gboost.fit(train_pca_df, new_y_train)

    print("Best parameters:", CV_gboost.best_params_)
    print("Best score:", CV_gboost.best_score_)

    return CV_gboost.best_params_
