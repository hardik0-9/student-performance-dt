import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. LOAD DATASET
DATA_PATH = "Student_Performance_DT.csv"

df = pd.read_csv(DATA_PATH)

print("\n DATASET PREVIEW")

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# 2. DATA PREPROCESSING
print("\n================ DATA PREPROCESSING ================")

# Convert Final Result into numerical values
df["Final_Result"] = df["Final_Result"].map({
    "Pass": 1,
    "Fail": 0
})

# Convert Internet Access into numerical values
df["Internet_Access"] = df["Internet_Access"].map({
    "Yes": 1,
    "No": 0
})

# Convert Extracurricular into numerical values
df["Extracurricular"] = df["Extracurricular"].map({
    "Yes": 1,
    "No": 0
})

print("\nDataset after encoding:")
print(df.head())

# 3. CORRELATION ANALYSIS
print("\n================ CORRELATION ANALYSIS ================")

corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(corr)

print("\nCorrelation with Final Result:")
print(
    corr["Final_Result"]
    .sort_values(ascending=False)
)

# 4. FEATURE AND TARGET SELECTION
print("\n================ FEATURE AND TARGET ================")

# Target variable
y = df["Final_Result"]

# Feature variables
x = df.drop("Final_Result", axis=1)

print("\nFeatures:")
print(x.columns.tolist())

print("\nTarget:")
print("Final_Result")

# 5. TRAIN-TEST SPLIT

print("\n================ TRAIN-TEST SPLIT ================")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features Shape:")
print(x_train.shape)

print("\nTesting Features Shape:")
print(x_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# 6. DEFAULT DECISION TREE CLASSIFIER
print("\n================ DEFAULT DECISION TREE ================")

default_model = DecisionTreeClassifier(
    random_state=42
)

# Train the model
default_model.fit(x_train, y_train)

# Make predictions
y_pred_default = default_model.predict(x_test)

print("\nDefault Model Predictions:")
print(y_pred_default)

# Calculate accuracy
default_acc = accuracy_score(
    y_test,
    y_pred_default
)

print("\nDefault Decision Tree Accuracy:")
print(f"{default_acc:.4f}")

# Confusion matrix
default_confusion = confusion_matrix(
    y_test,
    y_pred_default
)

print("\nDefault Decision Tree Confusion Matrix:")
print(default_confusion)

# 7. HYPERPARAMETER TUNING USING GRIDSEARCHCV
print("\n================ GRIDSEARCHCV ================")

grid_param = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [3, 5, 7]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(
        random_state=42
    ),
    param_grid=grid_param,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train GridSearchCV
grid_search.fit(x_train, y_train)

# 8. BEST PARAMETERS AND MODEL
print("\nBEST MODEL PARAMETERS")

print("\nBest Cross-Validation Score:")
print(f"{grid_search.best_score_:.4f}")

print("\nBest Parameters:")
print(grid_search.best_params_)

tuned_model = grid_search.best_estimator_

print("\nBest Tuned Decision Tree Model:")
print(tuned_model)

# 9. TUNED MODEL PREDICTION
print("\nTUNED MODEL PREDICTION")

# Make predictions using tuned model
y_pred_tuned = tuned_model.predict(x_test)

print("\nTuned Model Predictions:")
print(y_pred_tuned)

# Calculate tuned accuracy
tuned_acc = accuracy_score(
    y_test,
    y_pred_tuned
)

print("\nTuned Decision Tree Accuracy:")
print(f"{tuned_acc:.4f}")

# Tuned confusion matrix
tuned_confusion = confusion_matrix(
    y_test,
    y_pred_tuned
)

print("\nTuned Decision Tree Confusion Matrix:")
print(tuned_confusion)

# 10. MODEL COMPARISON
print("\nMODEL COMPARISON")

improvement = tuned_acc - default_acc

print(
    f"\nDefault Decision Tree Accuracy : "
    f"{default_acc:.4f}"
)

print(
    f"Tuned Decision Tree Accuracy   : "
    f"{tuned_acc:.4f}"
)

print(
    f"Accuracy Improvement           : "
    f"{improvement:.4f}"
)

print(
    f"Accuracy Improvement (%)       : "
    f"{improvement * 100:.2f}%"
)

# 11. FINAL RESULT
print("\nFINAL RESULT")

if tuned_acc > default_acc:
    print("Hyperparameter tuning improved the model.")
elif tuned_acc == default_acc:
    print("Hyperparameter tuning produced the same accuracy.")
else:
    print("Default model performed better than the tuned model.")

print(
    f"\nFinal Tuned Model Accuracy: "
    f"{tuned_acc:.2%}"
)

print("\n================================================")
print("Student Performance Prediction Completed!")
print("================================================")