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
# nan = sum(data.isnull().sum()) # 0 NaN values


# Select X and y from data
X = data.iloc[:, 2:].values
y = data["target"].values

# Positive/Negative ratio
# print((y == True).sum() / len(y)) # 0.10049 -> Around 1 out of 10 samples is positive


# Split data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=8, stratify=y)

# # Check whether the split has equitative positive and negative labels
# print((y_train == True).sum() / len(y_train)) # 0.1004875
# print((y_test == True).sum() / len(y_test)) # 0.1005


# Histogram of features mean values
def hist_mean_values(X=X):
    features_mean_values = np.mean(X, axis=0)
    plt.figure()
    plt.hist(features_mean_values)
    plt.show()

hist_mean_values(X)

# The features mean values has a Gaussian shape, maybe some features are correlated


# Histogram of features std values
def hist_std_values(X=X):
    features_std_values = np.std(X, axis=0)
    plt.figure()
    plt.hist(features_std_values)
    plt.show()

hist_std_values(X)

# The features std values are mostly close to 0, this could lead to the correlation hypothesis


# Plot 100 first values of the four least uniform features for insight purpose
def select_highest_std_features(X):
    features_std_values = np.std(X, axis=0)
    features_std_values_sorted_ix = np.argsort(features_std_values)[::-1]
    return X[:,features_std_values_sorted_ix[:4]]

def plot_four_features(X_four_features):
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(np.arange(100), X_four_features[:100,i])
        if (i == 0) or (i == 2):
            plt.ylabel('Var value')
        if (i > 1):
            plt.xlabel('Sample')
    plt.show()

plot_four_features(select_highest_std_features(X))


# Scale features
from sklearn.preprocessing import StandardScaler

def scale_data(X_train=X_train, X_test=X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

X_train_scaled, X_test_scaled = scale_data()


# Reduce train and test sets sample size to faster try out models
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X, y,
                                                    test_size=0.05, train_size=0.2,
                                                    random_state=8, stratify=y)

# Check whether the split has equitative positive and negative labels
# print((y_train_red == True).sum() / len(y_train_red)) # 0.1004875
# print((y_test_red == True).sum() / len(y_test_red)) # 0.1005

# Scale reduced X features
X_train_scaled_red, X_test_scaled_red = scale_data(X_train_red, X_test_red)


# Scale features using RobustScaler
from sklearn.preprocessing import RobustScaler

def scale_R_data(X_train=X_train, X_test=X_test):
    scaler_robust = RobustScaler()
    scaler_robust.fit(X_train)
    return scaler_robust.transform(X_train), scaler_robust.transform(X_test)


X_train_scaled_R, X_test_scaled_R = scale_R_data()


# Scale w/ robust reduced X features
X_train_scaled_R_red, X_test_scaled_R_red = scale_R_data(X_train_red, X_test_red)


# # Set classifiers
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


# Scores for SGD classifier using reduced and not-reduced X_train and standar/robust scaler

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

# sgd_clf_scores = clf_scores(sgd_clf, X_train_scaled_R, y_train, title="SGD (Robust)")
# # SGD (Robust)
# # Train Precision = 0.4738
# # Test Precision = 0.4551
# # Train Recall = 0.2819
# # Test Recall = 0.2770
# # Train F1 score = 0.3507
# # Test F1 score = 0.3419
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

def predict_scores_sgd(X=X_train_scaled, y_true=y_train,
                       sgd_clf=SGDClassifier(random_state=8)):
    y_scores_sgd = cross_val_predict(sgd_clf, X, y_true,
                                     cv=3, method='decision_function')
    return y_true, y_scores_sgd

def predict_scores_forest(X=X_train_scaled, y_true=y_train,
                          forest_clf=RandomForestClassifier(random_state=8)):
    y_scores_forest = cross_val_predict(forest_clf, X, y_true,
                                        cv=2, method='predict_proba')[:, 1]
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
# y_true, y_scores_sgd = predict_scores_sgd(X_train_scaled, y_train)
# plot_precision_recall_curve(y_true, y_scores_sgd, "Default SGD")
#
# y_true, y_scores_forest = predict_scores_forest(X_train_scaled, y_train)
# plot_precision_recall_curve(y_true, y_scores_forest, "Default forest")


# Tunning hyper-parameters
from sklearn.model_selection import GridSearchCV

# Grid search for classifier, given a parameter grid. Save results as DataFrame pickle in given path
# Return best estimator and DataFrame results
def hyper_parameter_tuning(clf, param_grid, scoring, df_path,
                           cv=3, refit_parameter=None,
                           X=X_train_scaled, y=y_train):
    grid_search = GridSearchCV(clf, param_grid, scoring, cv=cv,
                               refit=refit_parameter, return_train_score=True)
    grid_search.fit(X, y)
    results_grid_search = pd.DataFrame(grid_search.cv_results_)
    results_grid_search.to_csv(df_path)
    return grid_search.best_estimator_, results_grid_search


# Tuning SGD hyper-parameters

# Grid search for SGD classifier
def grid_search_sgd(cv=3, X=X_train_scaled, y=y_train):
    sgd_clf = SGDClassifier(random_state=8, penalty='elasticnet')
    param_grid_sgd = [{"loss": ["squared_hinge"],
                      "alpha": [0.001],
                      "l1_ratio": [0.4, 0.6, 0.9]},
                      {"loss": ["squared_hinge"],
                       "alpha": [0.01, 0.03, 0.1, 0.3],
                       "l1_ratio": [0.4, 0.6]},
                      {"loss": ["perceptron"],
                      "alpha": [0.0008, 0.002],
                      "l1_ratio": [0.7, 0.9]},
                      {"loss": ["perceptron"],
                       "alpha": [0.008, 0.02],
                       "l1_ratio": [0.1, 0.3]},
                      {"loss": ["hinge"],
                       "alpha": [0.008, 0.02],
                       "l1_ratio": [0.2, 0.4]},
                      {"loss": ["log"],
                       "alpha": [0.001],
                       "l1_ratio": [0.4, 0.6]},
                      {"loss": ["log"],
                       "alpha": [0.01],
                       "l1_ratio": [0.2, 0.4]}]
    # param_grid_sgd = {"alpha": [0.0001, 0.01]}
    scoring = ["precision", "recall", "f1"]
    df_path = "./GridSearch dataframes/SGD.csv"
    return hyper_parameter_tuning(sgd_clf, param_grid_sgd, scoring, df_path,
                                  cv=cv, refit_parameter="precision",
                                  X=X, y=y)


# Grid search and oobservations
# _, sgd_gs_results_4 = grid_search_sgd(2, X_train_scaled_red, y_train_red)
# 1 search: (alpha: best=0.1, discard:1)
# 2 search: (alpha: discard:0.1)
#           (loss: best recall/F1:"squared_hinge" w/ alpha=0.01,0.1,
#           best precision:"hinge","log" w/ alpha=0.01 w/ low l1)
# 3 search: (alpha: 0.1 only for "squared_hinge"),
#           ("squared_hinge": alpha=0.001 w/ high l1 alpha),
#           ("hinge": discard alpha=0.001, best l1=0.3),
#           ("perceptron": low alpha w/ high l1),
#           ("log": low alpha w/ high l1, high alpha w/low l1)
# 4 search: similar results


# # Chosen classifiers:
# sgd_clf_1 = SGDClassifier(random_state=8, penalty='elasticnet', loss="squared_hinge",
#                           alpha=0.01, l1_ratio=0.4)
# sgd_clf_2 = SGDClassifier(random_state=8, penalty='elasticnet', loss="perceptron",
#                           alpha=0.002, l1_ratio=0.7)
# sgd_clf_3 = SGDClassifier(random_state=8, penalty='elasticnet', loss="hinge",
#                           alpha=0.01, l1_ratio=0.3)
# sgd_clf_4 = SGDClassifier(random_state=8, penalty='elasticnet', loss="log",
#                           alpha=0.001, l1_ratio=0.6)


# Predict reduced y_scores and plot precision-recall curve for SGD classifiers
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_1, X_train_scaled_red, y_train_red)
# plot_precision_recall_curve(y_true, y_scores_sgd, "SGD 1")
#
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_2, X_train_scaled_red, y_train_red)
# plot_precision_recall_curve(y_true, y_scores_sgd, "SGD 2")
#
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_3, X_train_scaled_red, y_train_red)
# plot_precision_recall_curve(y_true, y_scores_sgd, "SGD 3")
#
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_4, X_train_scaled_red, y_train_red)
# plot_precision_recall_curve(y_true, y_scores_sgd, "SGD 4")


# Very similar behaviour. SGD 1 and 2 are more stable over threshold, SGD 2 is selected as the best SGD classifier
# sgd_clf_scores = clf_scores(sgd_clf_2, title="Tunned SGD classifier")
# Tunned SGD classifier
# Train Precision = 0.5860
# Test Precision = 0.5790   -> Improved 42.7% from default SGD classifier
# Train Recall = 0.3722
# Test Recall = 0.3673      -> Improved 24.1% from default SGD classifier
# Train F1 score = 0.4552
# Test F1 score = 0.4494    -> Improved 32.1% from default SGD classifier


# # Tested tuned SGD classifier with robust scaling
# sgd_clf_scores = clf_scores(sgd_clf_1, X_train_scaled_R, y_train, title="Tunned SGD classifier (Robust scaled)")
# # Tunned SGD classifier 1 (Robust scaled)
# # Train Precision = 0.5041
# # Test Precision = 0.5049
# # Train Recall = 0.4603
# # Test Recall = 0.4627
# # Train F1 score = 0.4812
# # Test F1 score = 0.4829


# Grid search for SGD classifier with robust scaling
def grid_search_sgd_robust(cv=3, X=X_train_scaled_R, y=y_train):
    sgd_clf = SGDClassifier(random_state=8, penalty='elasticnet')
    param_grid_sgd = [{"loss": ["squared_hinge"],
                      "alpha": [0.001],
                      "l1_ratio": [0.4, 0.6, 0.9]},
                      {"loss": ["squared_hinge"],
                       "alpha": [0.01, 0.03, 0.1, 0.3],
                       "l1_ratio": [0.4, 0.6]}]
    # param_grid_sgd = {"alpha": [0.0001, 0.01]}
    scoring = ["precision", "recall", "f1"]
    df_path = "./GridSearch dataframes/SGD_robust.csv"
    return hyper_parameter_tuning(sgd_clf, param_grid_sgd, scoring, df_path,
                                  cv=cv, refit_parameter="precision",
                                  X=X, y=y)

# _, sgd_gs_results_robust = grid_search_sgd_robust(2)
# # Same results as with standard scaling

# Plot learning curves of SGD classifier
from sklearn.metrics import f1_score

def plot_SGD_learning_curve(max_epochs=1000, max_iter_sgd=100):
    if max_iter_sgd > max_epochs:
        print("max_epochs must be greater or equal to max_iter_sgd")
    else:
        # Split reduced data into train and validation sets
        X_train_learn, X_val_learn, y_train_learn, y_val_learn = train_test_split(X_train_scaled_red, y_train_red,
                                                                                  test_size=0.2, random_state=8,
                                                                                  stratify=y_train_red)

        # Classifier with fixed max_iter to check F1 score over epochs
        sgd_clf_step = SGDClassifier(max_iter=max_iter_sgd, random_state=8, penalty='elasticnet', loss="squared_hinge",
                                     alpha=0.01, l1_ratio=0.4, warm_start=True, tol=-np.infty)

        # Create arrays of F1 scores over epochs
        n_epochs = int(max_epochs/max_iter_sgd)
        train_scores_array = np.empty(n_epochs)
        val_scores_array = np.empty(n_epochs)

        for epoch in range(n_epochs):
            # random_ix = np.random.permutation(len(y_train_learn))
            # sgd_clf_step.fit(X_train_learn[random_ix,:], y_train_learn.ravel()[random_ix])
            sgd_clf_step.fit(X_train_learn, y_train_learn.ravel())
            y_train_learn_predict = sgd_clf_step.predict(X_train_learn)
            y_val_learn_predict = sgd_clf_step.predict(X_val_learn)
            train_score = f1_score(y_train_learn, y_train_learn_predict)
            val_score = f1_score(y_val_learn, y_val_learn_predict)
            train_scores_array[epoch] = train_score
            val_scores_array[epoch] = val_score

        # Plot train and validation learning curves
        plt.plot(val_scores_array, "b-", linewidth=3, label="Validation set")
        plt.plot(train_scores_array, "r--", linewidth=2, label="Training set")
        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel(("Epoch x" + str(max_iter_sgd)), fontsize=14)
        plt.ylabel("F1 score", fontsize=14)
        plt.show()

# plot_SGD_learning_curve(100, 10) # After around 40 epochs, the SGD classifier converges. No need to tune learning rate