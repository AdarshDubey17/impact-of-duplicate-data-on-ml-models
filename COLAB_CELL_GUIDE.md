# Colab notebook — cell-by-cell copy guide
> The same cells are also available in `notebooks/duplicate_data_experiment.ipynb`. In Colab, uploading/opening that notebook is easier than pasting manually.

## Cell 1 — 01 - Project title (Markdown)

```markdown
# Impact of Duplicate Training Samples on Machine Learning Generalization

An experimental study of how adding different percentages of duplicate records to training data affects the performance and generalization of multiple machine learning classifiers.

**Dataset:** UCI Internet Firewall Data  
**Models:** Logistic Regression, Decision Tree, Random Forest and K-Nearest Neighbors (KNN)
```

## Cell 2 — 02 - Problem statement and objectives (Markdown)

```markdown
## Problem Statement

To investigate how increasing the number of duplicate training samples affects the predictive performance and generalization of different machine learning models.

## Objectives

- Compare four classification algorithms.
- Test duplicate levels of 0%, 10%, 25%, 50%, 75% and 100%.
- Evaluate models using multiple performance metrics.
- Examine training time and the training–testing generalization gap.
```

## Cell 3 — 03 - Dataset and methodology (Markdown)

```markdown
## Dataset

The project uses the UCI Internet Firewall Data dataset. It contains firewall traffic records with 11 input attributes and the target column `Action`, which has four classes: `allow`, `deny`, `drop` and `reset-both`. The original dataset contains 65,532 records. Exact duplicate rows are removed before the train-test split.

**Dataset source:** [UCI Internet Firewall Data](https://archive.ics.uci.edu/dataset/542/internet%2Bfirewall%2Bdata)  
**DOI:** `10.24432/C5131M`

## Experimental Method

The cleaned data is split into 80% training and 20% testing data using stratification. Duplicate rows are sampled with replacement from the training set only and added at six levels: 0%, 10%, 25%, 50%, 75% and 100% of the original training-set size. The four models are evaluated on the same untouched test set. The experiment is repeated with two duplicate-sampling seeds: 42 and 123.

The notebook reports accuracy, weighted precision, weighted recall, weighted F1, macro F1, ROC-AUC, generalization gap and training time.
```

## Cell 4 — 04 - Imports and setup (Code)

```python
!pip -q install pandas numpy scikit-learn matplotlib seaborn

import os
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from IPython.display import display

os.makedirs("project_outputs", exist_ok=True)
print("Libraries imported and output folder created.")
```

## Cell 5 — 05 - Load dataset (Code)

```python
url = "https://raw.githubusercontent.com/balarabetahir/Firewall-Log-Classification-with-Machine-Learning.ipynb/main/log2.csv"

df = pd.read_csv(url)
df.columns = df.columns.str.strip()
df["Action"] = df["Action"].astype(str).str.strip().str.lower()

print("DATASET LOADED")
print("Original shape:", df.shape)
print("\nFirst five records:")
display(df.head())
```

## Cell 6 — 06 - Explore dataset (Code)

```python
print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

original_duplicates = df.duplicated().sum()
print("\nExact duplicate records:", original_duplicates)

print("\nOriginal class distribution:")
print(df["Action"].value_counts())

# Class distribution visualization
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Action", order=df["Action"].value_counts().index)
plt.title("Firewall Action Class Distribution")
plt.xlabel("Firewall Action")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()
```

## Cell 7 — 07 - Clean and preprocess (Code)

```python
# Remove pre-existing exact duplicate records before splitting
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop("Action", axis=1).apply(pd.to_numeric, errors="coerce")
y_text = df["Action"]

# Handle any missing feature values safely
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)
class_names = encoder.classes_

print("\nCleaned dataset shape:", df.shape)
print("Features:", X.shape[1])
print("Classes:", list(class_names))
print("Remaining exact duplicates:", df.duplicated().sum())

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
```

## Cell 8 — 08 - Train-test split (Code)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
```

## Cell 9 — 09 - Create models (Code)

```python
def create_models():

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ))
        ])
    }
```

## Cell 10 — 10 - Run duplicate-data experiment (Code)

```python
duplicate_levels = [0, 10, 25, 50, 75, 100]
seeds = [42, 123]

results = []
reports = []
confusion_data = []
feature_importance = None

for seed in seeds:

    print("\nRANDOM SEED:", seed)

    for level in duplicate_levels:

        n_duplicates = int(len(X_train) * level / 100)

        rng = np.random.RandomState(seed)

        if n_duplicates > 0:

            indices = rng.choice(
                len(X_train),
                size=n_duplicates,
                replace=True
            )

            X_dup = X_train.iloc[indices]
            y_dup = y_train[indices]

            X_exp = pd.concat(
                [X_train, X_dup],
                ignore_index=True
            )

            y_exp = np.concatenate([y_train, y_dup])

        else:

            X_exp = X_train.reset_index(drop=True).copy()
            y_exp = y_train.copy()

        print(
            f"\nDuplicates: {level}% | "
            f"Training records: {len(X_exp)}"
        )

        for model_name, model in create_models().items():

            start = time.time()
            model.fit(X_exp, y_exp)
            training_time = time.time() - start

            train_pred = model.predict(X_exp)
            test_pred = model.predict(X_test)
            probabilities = model.predict_proba(X_test)

            train_acc = accuracy_score(y_exp, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            precision = precision_score(
                y_test, test_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test, test_pred,
                average="weighted",
                zero_division=0
            )

            f1_weighted = f1_score(
                y_test, test_pred,
                average="weighted",
                zero_division=0
            )

            f1_macro = f1_score(
                y_test, test_pred,
                average="macro",
                zero_division=0
            )

            roc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="weighted",
                labels=np.arange(len(class_names))
            )

            gap = train_acc - test_acc

            results.append({
                "Seed": seed,
                "Duplicate Level": level,
                "Training Samples": len(X_exp),
                "Model": model_name,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
                "Precision": precision,
                "Recall": recall,
                "Weighted F1": f1_weighted,
                "Macro F1": f1_macro,
                "ROC-AUC": roc,
                "Generalization Gap": gap,
                "Training Time": training_time
            })

            # Save detailed class-wise report for baseline and 100%
            if seed == 42 and level in [0, 100]:

                report = classification_report(
                    y_test,
                    test_pred,
                    labels=np.arange(len(class_names)),
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0
                )

                for class_name, values in report.items():
                    if isinstance(values, dict):
                        reports.append({
                            "Duplicate Level": level,
                            "Model": model_name,
                            "Class": class_name,
                            **values
                        })

                cm = confusion_matrix(
                    y_test,
                    test_pred,
                    labels=np.arange(len(class_names))
                )

                confusion_data.append({
                    "Duplicate Level": level,
                    "Model": model_name,
                    "Matrix": cm
                })

                plt.figure(figsize=(7,5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names
                )
                plt.title(f"{model_name} - {level}% Duplicates")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.tight_layout()

                plt.savefig(
                    f"project_outputs/confusion_{model_name.replace(' ','_')}_{level}.png",
                    dpi=250
                )
                plt.show()
                plt.close()

            if (
                seed == 42
                and level == 0
                and model_name == "Random Forest"
            ):

                feature_importance = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)

            print(
                f"{model_name}: "
                f"Accuracy={test_acc:.4f}, "
                f"F1={f1_weighted:.4f}, "
                f"Time={training_time:.2f}s"
            )
```

## Cell 11 — 11 - Save and average results (Code)

```python
results_df = pd.DataFrame(results)

results_df.to_csv(
    "project_outputs/all_experiment_results.csv",
    index=False
)

results_df[results_df["Seed"] == 42].to_csv(
    "project_outputs/experiment_results_42.csv",
    index=False
)

results_df[results_df["Seed"] == 123].to_csv(
    "project_outputs/experiment_results_123.csv",
    index=False
)

reports_df = pd.DataFrame(reports)
reports_df.to_csv(
    "project_outputs/classification_reports.csv",
    index=False
)

average_results = (
    results_df
    .drop(columns=["Seed"])
    .groupby(
        ["Duplicate Level", "Model"],
        as_index=False
    )
    .mean(numeric_only=True)
)

average_results.to_csv(
    "project_outputs/average_results.csv",
    index=False
)

print("\nALL EXPERIMENT RESULTS")
display(average_results.round(4))
```

## Cell 12 — 12 - Original vs duplicated vs deduplicated comparison (Code)

```python
# ================================================================
# 8. ORIGINAL VS DUPLICATED VS DEDUPLICATED COMPARISON
# ================================================================

# Compare original, deliberately duplicated and deduplicated
# training datasets using the same held-out test set.

comparison_data = []

# Baseline: original training dataset
comparison_sets = {
    "Original": (
        X_train.reset_index(drop=True),
        y_train.copy()
    )
}

# Add 100% duplicates
rng = np.random.RandomState(42)
indices = rng.choice(
    len(X_train),
    size=len(X_train),
    replace=True
)

X_dup = X_train.iloc[indices]
y_dup = y_train[indices]

comparison_sets["Duplicated"] = (
    pd.concat([X_train, X_dup], ignore_index=True),
    np.concatenate([y_train, y_dup])
)

# Deduplicate the duplicated training set
combined = X_dup.copy()
combined["__target__"] = y_dup

original_part = X_train.copy()
original_part["__target__"] = y_train

combined = pd.concat(
    [original_part, combined],
    ignore_index=True
).drop_duplicates()

X_dedup = combined.drop(columns="__target__")
y_dedup = combined["__target__"].to_numpy()

comparison_sets["Deduplicated"] = (
    X_dedup,
    y_dedup
)

for dataset_name, (X_set, y_set) in comparison_sets.items():

    for model_name, model in create_models().items():

        model.fit(X_set, y_set)
        pred = model.predict(X_test)

        comparison_data.append({
            "Training Dataset": dataset_name,
            "Model": model_name,
            "Training Samples": len(X_set),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Weighted F1": f1_score(
                y_test, pred,
                average="weighted",
                zero_division=0
            )
        })

comparison_df = pd.DataFrame(comparison_data)

comparison_df.to_csv(
    "project_outputs/deduplication_comparison.csv",
    index=False
)

print("\nORIGINAL VS DUPLICATED VS DEDUPLICATED")
display(comparison_df.round(4))
```

## Cell 13 — 13 - Feature importance (Code)

```python
# ================================================================
# 9. FEATURE IMPORTANCE
# ================================================================

if feature_importance is not None:

    feature_importance.to_csv(
        "project_outputs/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=feature_importance.head(12),
        x="Importance",
        y="Feature"
    )
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(
        "project_outputs/feature_importance.png",
        dpi=250
    )
    plt.show()
    plt.close()
```

## Cell 14 — 14 - Performance visualizations (Code)

```python
# 10. PERFORMANCE VISUALIZATION FUNCTION
# ================================================================

metrics = [
    ("Test Accuracy", "Duplicate Level vs Test Accuracy"),
    ("Weighted F1", "Duplicate Level vs Weighted F1 Score"),
    ("Macro F1", "Duplicate Level vs Macro F1 Score"),
    ("ROC-AUC", "Duplicate Level vs ROC-AUC"),
    ("Generalization Gap", "Duplicate Level vs Generalization Gap"),
    ("Training Time", "Duplicate Level vs Training Time")
]

for metric, title in metrics:

    plt.figure(figsize=(10,6))

    for model_name in average_results["Model"].unique():

        selected = average_results[
            average_results["Model"] == model_name
        ]

        plt.plot(
            selected["Duplicate Level"],
            selected[metric],
            marker="o",
            label=model_name
        )

    plt.title(title)
    plt.xlabel("Duplicate Level (%)")
    plt.ylabel(metric)
    plt.xticks(duplicate_levels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = metric.lower().replace(" ", "_") + ".png"

    plt.savefig(
        f"project_outputs/{filename}",
        dpi=250
    )

    plt.show()
    plt.close()
```

## Cell 15 — 15 - Baseline model comparison (Code)

```python
# ================================================================
# 11. MODEL COMPARISON BAR CHART
# ================================================================

baseline = average_results[
    average_results["Duplicate Level"] == 0
]

plt.figure(figsize=(9,5))
sns.barplot(
    data=baseline,
    x="Model",
    y="Test Accuracy"
)
plt.title("Baseline Model Accuracy Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    "project_outputs/baseline_model_comparison.png",
    dpi=250
)
plt.show()
plt.close()
```

## Cell 16 — 16 - Class-wise performance (Code)

```python
# ================================================================
# 12. CLASS-WISE PERFORMANCE VISUALIZATION
# ================================================================

if not reports_df.empty:

    class_reports = reports_df[
        reports_df["Class"].isin(class_names)
    ]

    for metric in ["precision", "recall", "f1-score"]:

        plt.figure(figsize=(11,6))

        sns.barplot(
            data=class_reports[class_reports["Duplicate Level"] == 0],
            x="Class",
            y=metric,
            hue="Model"
        )

        plt.title(f"Class-wise {metric.title()} - Baseline")
        plt.tight_layout()
        plt.savefig(
            f"project_outputs/classwise_{metric}.png",
            dpi=250
        )
        plt.show()
        plt.close()
```

## Cell 17 — 17 - Model ranking and summary (Code)

```python
# ================================================================
# 13. FINAL MODEL RANKING
# ================================================================

ranking = baseline.sort_values(
    "Test Accuracy",
    ascending=False
)[["Model", "Test Accuracy", "Weighted F1", "ROC-AUC"]]

print("\nBASELINE MODEL RANKING")
display(ranking.round(4))

# ================================================================
# 14. AUTOMATIC RESULT SUMMARY
# ================================================================

print("\n" + "=" * 65)
print("FINAL MINI PROJECT SUMMARY")
print("=" * 65)

print("Dataset records after cleaning:", len(df))
print("Original exact duplicates:", original_duplicates)
print("Number of features:", X.shape[1])
print("Number of classes:", len(class_names))
print("Classes:", list(class_names))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Duplicate levels:", duplicate_levels)
print("Models:", list(create_models().keys()))
print("Random seeds:", seeds)
print("Total model experiments:", len(results_df))

print("\nBest baseline model:")
print(ranking.iloc[0].to_string())

print("\nAll files saved inside project_outputs folder.")
```

## Cell 18 — 18 - Package and download outputs (Code)

```python
# ================================================================
# 15. ZIP ALL PROJECT OUTPUTS
# ================================================================

with zipfile.ZipFile(
    "mini_project_outputs.zip",
    "w",
    zipfile.ZIP_DEFLATED
) as zip_file:

    for root, dirs, files in os.walk("project_outputs"):
        for file in files:
            path = os.path.join(root, file)
            zip_file.write(path)

print("\nZIP FILE CREATED: mini_project_outputs.zip")

from google.colab import files
files.download("mini_project_outputs.zip")
```
