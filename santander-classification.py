import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Load dataset
SANTANDER_PATH = "data"

def load_santander_data(santander_path=SANTANDER_PATH):
    csv_path = os.path.join(santander_path, "train.csv")
    return pd.read_csv(csv_path)

data = load_santander_data()

# data.head()
# data.info()
# Dataset contains:
# 1 ID column, 1 target column and 200 feature columns.
# 200000 samples


# Check for NaN values
nan = sum(data.isnull().sum()) # 0 NaN values


# Select X and y from data
X = data.iloc[:, 2:].values
y = data["target"].values

# Positive/Negative ratio
# print((y == True).sum() / len(y)) # 0.10049 -> Around 1 out of 10 samples is positive


# Split data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=8, stratify=y)

# Check whether the split has equitative positive and negative labels
# print((y_train == True).sum() / len(y_train)) # 0.1004875
# print((y_test == True).sum() / len(y_test)) # 0.1005


# Scale features
from sklearn.preprocessing import StandardScaler

def scale_data(X_train=X_train, X_test=X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

X_train_scaled, X_test_scaled = scale_data()


# Reduce train and test sets sample size to faster try out models
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X, y,
                                                    test_size=0.1, train_size=0.4,
                                                    random_state=8, stratify=y)

# Check whether the split has equitative positive and negative labels
# print((y_train_red == True).sum() / len(y_train_red)) # 0.1004875
# print((y_test_red == True).sum() / len(y_test_red)) # 0.1005

# Scale reduced X features
X_train_scaled_red, X_test_scaled_red = scale_data(X_train_red, X_test_red)


# Set classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

sgd_clf = SGDClassifier(random_state=8)
forest_clf = RandomForestClassifier(random_state=8)


# Print cross-validate precision, recall and F1 score for classifier
from sklearn.model_selection import cross_validate

def clf_scores(clf, X=X_train_scaled, y=y_train, title=None):
    scoring = ["precision", "recall", "f1"]
    scores = cross_validate(clf, X, y,
                            cv=3, scoring=scoring)
    train_precision = np.mean(scores["train_precision"])
    train_recall = np.mean(scores["train_recall"])
    train_f1 = np.mean(scores["train_f1"])
    test_precision = np.mean(scores["test_precision"])
    test_recall = np.mean(scores["test_recall"])
    test_f1 = np.mean(scores["test_f1"])
    if title:
        print(title)
    print(f"Train Precision = {train_precision:.4f}")
    print(f"Test Precision = {test_precision:.4f}")
    print(f"Train Recall = {train_recall:.4f}")
    print(f"Test Recall = {test_recall:.4f}")
    print(f"Train F1 score = {train_f1:.4f}")
    print(f"Test F1 score = {test_f1:.4f}")
    return scores


# Scores for SGD classifier using reduced and not-reduced X_train

# sgd_clf_scores = clf_scores(sgd_clf, X_train_scaled_red, y_train_red, title="SGD (reduced)")
# # SGD (reduced)
# # Train Precision = 0.3882
# # Test Precision = 0.3869
# # Train Recall = 0.3332
# # Test Recall = 0.3364
# # Train F1 score = 0.3584
# # Test F1 score = 0.3594
# # High bias
#
# sgd_clf_scores = clf_scores(sgd_clf, title="SGD (not-reduced)")
# # SGD (not-reduced)
# # Train Precision = 0.4187
# # Test Precision = 0.4057
# # Train Recall = 0.3034
# # Test Recall = 0.2960
# # Train F1 score = 0.3500
# # Test F1 score = 0.3403
# # High bias


# Scores for random forest classifier using reduced and not-reduced X_train

# forest_clf_scores = clf_scores(forest_clf, X_train_scaled_red, y_train_red, title="Forest (reduced)")
# # Forest (reduced)
# # Train Precision = 0.9999
# # Test Precision = 0.5301
# # Train Recall = 0.8446
# # Test Recall = 0.0129
# # Train F1 score = 0.9157
# # Test F1 score = 0.0253
# # High variance

# forest_clf_scores = clf_scores(forest_clf, title="Forest (not-reduced)")
# # Forest (not-reduced)
# # Train Precision = 1.0000
# # Test Precision = 0.5542
# # Train Recall = 0.8506
# # Test Recall = 0.0156
# # Train F1 score = 0.9192
# # Test F1 score = 0.0304
# # High variance


# Compute y scores (y_proba) for classifiers
from sklearn.model_selection import cross_val_predict

def predict_scores_sgd(X=X_train_scaled, y_true=y_train):
    y_scores_sgd = cross_val_predict(sgd_clf, X, y_true,
                                     cv=3, method='decision_function')
    return y_true, y_scores_sgd

def predict_scores_forest(X=X_train_scaled, y_true=y_train):
    y_scores_forest = cross_val_predict(forest_clf, X, y_true,
                                        cv=3, method='predict_proba')[:, 1]
    return y_true, y_scores_forest


# Plot precision-recall curve function
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(y_true, y_scores, title=None):
    plt.figure()
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(thresholds, precisions[:-1], 'b-', label="Precision")
    plt.plot(thresholds, recalls[:-1], 'g--', label="Recall")
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylim([0, 1])
    plt.xlim([min(thresholds), max(thresholds)])
    if title:
        plt.title(title)
    plt.show()


# Predict y_scores and plot precision-recall curve for classifiers
y_true, y_scores_sgd = predict_scores_sgd(X_train_scaled, y_train)
plot_precision_recall_curve(y_true, y_scores_sgd, "SGD")

y_true, y_scores_forest = predict_scores_forest(X_train_scaled, y_train)
plot_precision_recall_curve(y_true, y_scores_forest, "Forest")