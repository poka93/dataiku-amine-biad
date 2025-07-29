import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import lightgbm as lgb
import joblib
from sklearn.model_selection import GridSearchCV

# Load the data
try:
    train_df = pd.read_csv("data/census_income_learn.csv", header=None)
    test_df = pd.read_csv("data/census_income_test.csv", header=None)
except FileNotFoundError:
    print("Ensure the data files are in the 'data' directory.")
    exit()

# Add column names based on metadata
column_names = [
    "age",
    "class of worker",
    "detailed industry recode",
    "detailed occupation recode",
    "education",
    "wage per hour",
    "enroll in edu inst last wk",
    "marital stat",
    "major industry code",
    "major occupation code",
    "race",
    "hispanic origin",
    "sex",
    "member of a labor union",
    "reason for unemployment",
    "full or part time employment stat",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "tax filer stat",
    "region of previous residence",
    "state of previous residence",
    "detailed household and family stat",
    "detailed household summary in household",
    "instance weight",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "live in this house 1 year ago",
    "migration prev res in sunbelt",
    "num persons worked for employer",
    "family members under 18",
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "citizenship",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
    "weeks worked in year",
    "year",
    "income",
]

train_df.columns = column_names
test_df.columns = column_names

# Preprocessing
# Identify categorical and numerical features
categorical_features = train_df.select_dtypes(include=["object"]).columns
numerical_features = train_df.select_dtypes(include=np.number).columns

# Drop the target variable and the instance weight from the feature lists
categorical_features = categorical_features.drop("income")
numerical_features = numerical_features.drop(["instance weight"])

# Create the preprocessing pipelines for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)

# Define the target variable
X = train_df.drop("income", axis=1)
y = train_df["income"].apply(lambda x: 1 if x.strip() == "50000+." else 0)

X_test = test_df.drop("income", axis=1)
y_test = test_df["income"].apply(lambda x: 1 if x.strip() == "50000+." else 0)

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Baseline Model: Logistic Regression ---
print("Training Logistic Regression model...")
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                random_state=42, solver="liblinear", class_weight="balanced"
            ),
        ),
    ]
)

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict_proba(X_val)[:, 1]
roc_auc_lr = roc_auc_score(y_val, y_pred_lr)
f1_lr = f1_score(y_val, lr_pipeline.predict(X_val))
precision_lr, recall_lr, _ = precision_recall_curve(y_val, y_pred_lr)
pr_auc_lr = auc(recall_lr, precision_lr)

print(f"Logistic Regression Validation ROC AUC: {roc_auc_lr:.4f}")
print(f"Logistic Regression Validation F1 Score: {f1_lr:.4f}")
print(f"Logistic Regression Validation PR AUC: {pr_auc_lr:.4f}")

# --- Advanced Model: LightGBM with Hyperparameter Tuning ---
print("\nTraining LightGBM model with Hyperparameter Tuning...")

# Define the parameter grid
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__num_leaves": [31, 61],
    "classifier__reg_alpha": [0.1, 0.5],
}

# Create the LightGBM pipeline
lgb_pipeline_tuned = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", lgb.LGBMClassifier(random_state=42, class_weight="balanced")),
    ]
)

# Create the GridSearchCV object
grid_search = GridSearchCV(
    lgb_pipeline_tuned,
    param_grid,
    cv=3,
    scoring="average_precision",
    n_jobs=-1,
    verbose=2,
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best estimator
best_lgb_model = grid_search.best_estimator_

# Evaluate the tuned model
y_pred_lgb_tuned = best_lgb_model.predict_proba(X_val)[:, 1]
roc_auc_lgb_tuned = roc_auc_score(y_val, y_pred_lgb_tuned)
f1_lgb_tuned = f1_score(y_val, best_lgb_model.predict(X_val))
precision_lgb_tuned, recall_lgb_tuned, _ = precision_recall_curve(
    y_val, y_pred_lgb_tuned
)
pr_auc_lgb_tuned = auc(recall_lgb_tuned, precision_lgb_tuned)

print("\nBest LightGBM Hyperparameters:", grid_search.best_params_)
print(f"Tuned LightGBM Validation ROC AUC: {roc_auc_lgb_tuned:.4f}")
print(f"Tuned LightGBM Validation F1 Score: {f1_lgb_tuned:.4f}")
print(f"Tuned LightGBM Validation PR AUC: {pr_auc_lgb_tuned:.4f}")

# --- Final Evaluation on Test Set ---
print("\nEvaluating the best model on the test set...")
best_model = best_lgb_model if pr_auc_lgb_tuned > pr_auc_lr else lr_pipeline

y_pred_test = best_model.predict_proba(X_test)[:, 1]
roc_auc_test = roc_auc_score(y_test, y_pred_test)
f1_test = f1_score(y_test, best_model.predict(X_test))
precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_test)
pr_auc_test = auc(recall_test, precision_test)

print(f"Best Model Test ROC AUC: {roc_auc_test:.4f}")
print(f"Best Model Test F1 Score: {f1_test:.4f}")
print(f"Best Model Test PR AUC: {pr_auc_test:.4f}")

# --- Save the best model ---
print("\nSaving the best model...")
joblib.dump(best_model, "best_model.pkl")
print("Model saved as best_model.pkl")

# --- Feature Importance ---
if hasattr(best_model.named_steps["classifier"], "feature_importances_"):
    print("\nFeature Importances:")

    try:
        # Get feature names from the column transformer
        all_feature_names = best_model.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        importances = best_model.named_steps["classifier"].feature_importances_

        feature_importance_df = pd.DataFrame(
            {"feature": all_feature_names, "importance": importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="importance", ascending=False
        )

        print(feature_importance_df.head(15))

    except Exception as e:
        print(f"Could not get feature importances: {e}")
        # Fallback to just showing importances if names are mismatched
        importances = best_model.named_steps["classifier"].feature_importances_
        print("Top 15 importances (without names):")
        print(np.sort(importances)[::-1][:15])
