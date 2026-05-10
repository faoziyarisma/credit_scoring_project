from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib

# Source link hyperparameter tuning: https://docs.google.com/document/d/1VX9jaLjYAfpJNPIcLjSrA1isyTevH_j55ib6auyhUYM/edit?usp=sharing


def decision_tree_with_optimal_hyperparam(train_pca_df, new_y_train):
    tree_model = DecisionTreeClassifier(
        random_state=123, criterion="entropy", max_depth=8, max_features="auto"
    )

    tree_model.fit(train_pca_df, new_y_train)
    print("score train:", tree_model.score(train_pca_df, new_y_train))
    joblib.dump(tree_model, "model/tree_model.joblib")
    return tree_model


def random_forest_with_optimal_hyperparam(train_pca_df, new_y_train):
    rdf_model = RandomForestClassifier(
        random_state=123,
        max_depth=8,
        n_estimators=200,
        max_features="sqrt",
        criterion="gini",
        n_jobs=-1,
    )
    rdf_model.fit(train_pca_df, new_y_train)
    print("score train:", rdf_model.score(train_pca_df, new_y_train))
    joblib.dump(rdf_model, "model/rdf_model.joblib")
    return rdf_model


def gradient_boosting_with_optimal_hyperparam(train_pca_df, new_y_train):
    gboost_model = GradientBoostingClassifier(
        random_state=123,
        learning_rate=0.1,
        max_depth=8,
        max_features="sqrt",
        n_estimators=200,
    )
    gboost_model.fit(train_pca_df, new_y_train)
    print("score train:", gboost_model.score(train_pca_df, new_y_train))
    joblib.dump(gboost_model, "model/gboost_model.joblib")
    return gboost_model
