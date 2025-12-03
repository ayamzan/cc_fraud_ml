import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import average_precision_score, make_scorer, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
# imbalance data handling
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
# classifiers for random forest, xgb, log regression, lgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import joblib

df = pd.read_csv('creditcard.csv')
print(df.head(5))
print(df.columns)
print(df.info)

# check for class imbalances
print(df.Class.value_counts())

sns.countplot(x="Class", data=df)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.histplot(df['Time'], bins=50, kde=True)
plt.subplot(1, 2, 2)
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of transaction amount')
plt.yscale('log')
plt.show()

# 0 not fraud, 1 is fraud
x = df.drop("Class", axis=1)
y = df["Class"]
x.head()
y.head()

print(f"\nfeatures shape: {x.shape}")
print(f"target shape: {y.shape}")

#stratified sampling to ensure fair proportion of data in both sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(f"x_train_shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("Training set class distribution")
print(y_train.value_counts(normalize=True))
print("\nTest set class distribution")
print(y_test.value_counts(normalize=True))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Time', 'Amount'])
    ],
    remainder='passthrough'
)

np.random.seed(42)

# calculate scaled weight for imbalance
scale_pos_weight_val = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"scale_pos_weight_val: {scale_pos_weight_val}")

random_state_val = 42
# define ColumnTransformer for preprocessing applying StandardScalar to TIme and Amount column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Time', 'Amount'])
    ],
    remainder='passthrough' # all other columns are kept as they are
)

# pipeline1: RandomForest + SMOTE (oversmapling)
pipeline_rf_smote = ImbPipeline([
    ('preprocessor', preprocessor), # ColumnTransformer applied
    ('smote', SMOTE(random_state=random_state_val)),
    ('classifier', RandomForestClassifier(random_state=random_state_val))
])
print("pl1: rf + smote")

# pipeline2: XGBoost with scaled pos weight (cost-sensitive learning)
pipeline_xgb_cw = ImbPipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(random_state=random_state_val,
                                    use_label_encoder=False,
                                    eval_metric='logloss',
                                    scale_pos_weight=scale_pos_weight_val))
])
print("pl2: xgb + scale pos weight")

# pipeline3: log regression with class weight (cost sensitive learning / baseline)
pipeline_lr_cw = ImbPipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=random_state_val,
                                      solver='liblinear',
                                      class_weight='balanced'))
])
print("pl3: log regression + class weight")

# pipeline4: lgbm with scale pos weight (cost-sensitive learning)
pipeline_lgb_cw = ImbPipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(random_state=random_state_val,
                                      scale_pos_weight=scale_pos_weight_val))
])
print("pl4: lgbm + scale pos weight]")

pipelines = {
    "RF + SMOTE": pipeline_rf_smote,
    "XGB + scale pos weight": pipeline_xgb_cw,
    "LR + class weight": pipeline_lr_cw,
    "LGB + scale pos weight": pipeline_lgb_cw
}
print("all pipelines created")

def evaluate_models(pipelines, x_train, y_train, x_test, y_test):
    results = []

    for name, pipeline in pipelines.items():
        print(f"\n training and evaluation: {name}")

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        # get predicted probabilities for the positive class (label 1)
        # many classifiers return shape (n_samples, n_classes)
        try:
            y_prob = pipeline.predict_proba(x_test)[:, 1]
        except Exception:
            # fallback: some estimators don't implement predict_proba
            # use decision_function if available and scale it for ranking metrics
            try:
                y_scores = pipeline.decision_function(x_test)
                # map scores to [0,1] via min-max for metrics that expect probabilities
                y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
            except Exception:
                # final fallback: use predicted labels as 0/1 probabilities
                y_prob = y_pred

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)

        results.append({
            "Model": name,
            "F1 score": f1,
            "ROC AUC": roc_auc,
            "Average Precision": avg_precision
        })
    # print per-model results so long runs show progress
    print(f"Completed: {name} -> F1={f1:.4f}, ROC_AUC={roc_auc:.4f}, AvgPrecision={avg_precision:.4f}")

    # ensure consistent column names and sort by F1 score (descending)
    df_res = pd.DataFrame(results)
    # unify column name casing if necessary
    if "F1 score" in df_res.columns and "F1 Score" not in df_res.columns:
        df_res = df_res.rename(columns={"F1 score": "F1 Score"})
    return df_res.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

results_df = evaluate_models(pipelines, x_train, y_train, x_test, y_test)
print("\nPerformance summary: ")
print(results_df)
# save trained pipeline objects to separate files (use .pkl filenames)
try:
    joblib.dump(pipelines["XGB + scale pos weight"], r"C:\Users\zantan\Downloads\cc_fraud\xgb_pipeline.pkl")
    joblib.dump(pipelines["RF + SMOTE"], r"C:\Users\zantan\Downloads\cc_fraud\rf_smote_pipeline.pkl")
    joblib.dump(pipelines["LR + class weight"], r"C:\Users\zantan\Downloads\cc_fraud\lr_pipeline.pkl")
    joblib.dump(pipelines["LGB + scale pos weight"], r"C:\Users\zantan\Downloads\cc_fraud\lgb_pipeline.pkl")
except Exception as e:
    print(f"Warning: failed to joblib.dump pipelines: {e}")

# fitting best model dynamically from results_df first row
best_model_key = None
# robustly get the top model name from results_df
if isinstance(results_df, pd.DataFrame) and not results_df.empty:
    if 'Model' in results_df.columns:
        # use loc to get the Model value in the first row
        best_model_key = results_df.iloc[0]["Model"]
    else:
        # fallback: take the value from the first cell of the first row
        best_model_key = results_df.iloc[0, 0]
else:
    print("Warning: results_df is empty or not a DataFrame; cannot determine best model key.")

if best_model_key and best_model_key in pipelines:
    best_model = pipelines[best_model_key]
else:
    print(f"Could not determine best model from results_df (got: {best_model_key}); defaulting to 'XGB + scale pos weight'.")
    best_model = pipelines.get("XGB + scale pos weight")

# Debug prints to show selection
print('results_df:')
print(results_df)
print(f"\nbest_model_key: {best_model_key}")
if best_model_key in pipelines:
    print(f"Selected pipeline object for: {best_model_key}")
else:
    print(f"Selected default pipeline: {'XGB + scale pos weight'}")

ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(best_model, x_test, y_test)
plt.title("ROC Curve")
plt.show()

PrecisionRecallDisplay.from_estimator(best_model, x_test, y_test)
plt.title("precision-Recall curve")
plt.show()