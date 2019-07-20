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
# print((y == True).sum() / len(y))
# 0.10049 -> Around 1 out of 10 samples is positive. Accuracy is not the best scoring parameter



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

# hist_mean_values(X)

# The features mean values has a Gaussian shape, maybe some features are correlated


# Histogram of features std values
def hist_std_values(X=X):
    features_std_values = np.std(X, axis=0)
    plt.figure()
    plt.hist(features_std_values)
    plt.show()

# hist_std_values(X)

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

# plot_four_features(select_highest_std_features(X))


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

# Scale w/ robust X features
# X_train_scaled_R, X_test_scaled_R = scale_R_data()


# Scale w/ robust reduced X features
# X_train_scaled_R_red, X_test_scaled_R_red = scale_R_data(X_train_red, X_test_red)


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


# Scores for SGD classifier using standard/robust scaling

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


# Scores for random forest classifier

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

def predict_scores_sgd(sgd_clf=SGDClassifier(random_state=8), X=X_train_scaled, y_true=y_train):
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
def hyper_parameter_tuning(clf, param_grid, scoring, df_path, cv=3, refit_parameter=None, X=X_train_scaled, y=y_train):
    grid_search = GridSearchCV(clf, param_grid, scoring, cv=cv,
                               refit=refit_parameter, return_train_score=True)
    grid_search.fit(X, y)
    drop_columns = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params',
                    'split0_test_precision','split1_test_precision', 'std_test_precision', 'split0_train_precision',
                    'split1_train_precision', 'std_train_precision', 'split0_test_recall', 'split1_test_recall',
                    'std_test_recall', 'split0_train_recall', 'split1_train_recall', 'std_train_recall', 'split0_test_f1',
                    'split1_test_f1', 'std_test_f1', 'split0_train_f1', 'split1_train_f1', 'std_train_f1']
    results_grid_search = pd.DataFrame(grid_search.cv_results_)
    results_grid_search.drop(columns=drop_columns, inplace=True)
    results_grid_search.to_csv(df_path)
    return grid_search.best_estimator_, results_grid_search


# Tuning SGD hyper-parameters

# Grid search for SGD classifier
def grid_search_sgd(param_grid, name, cv=3, X=X_train_scaled, y=y_train):
    sgd_clf = SGDClassifier(random_state=8, penalty='elasticnet')
    scoring = ["precision", "recall", "f1"]
    df_path = "./GridSearch dataframes/SGD_" + name + ".csv"
    return hyper_parameter_tuning(sgd_clf, param_grid, scoring, df_path, cv=cv, refit_parameter="precision", X=X, y=y)


# # Grid search for hinge loss (SVC)
# param_grid_sgd_hinge = {"loss": ["hinge"], "alpha": [0.0000001, 0.0000003, 0.000001, 0.000003, 0.00001, 0.00003,
#                                                      0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03,
#                                                      0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
#                         "l1_ratio": [0.2, 0.8]}
# _, sgd_gs_results_hinge = grid_search_sgd(param_grid_sgd_hinge, "hinge", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.8
#
# param_grid_sgd_hinge = {"loss": ["hinge"], "alpha": [0.0005, 0.001, 0.002], "l1_ratio": [0.6, 0.8, 0.9]}
# _, sgd_gs_results_hinge = grid_search_sgd(param_grid_sgd_hinge, "hinge_2", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.9
sgd_clf_hinge = SGDClassifier(random_state=8, penalty='elasticnet', alpha=0.001, l1_ratio=0.9)
# Precision = 0.59, Recall = 0.37, F1 score = 0.46


# # Grid search for log loss (Logistic regression)
# param_grid_sgd_log = {"loss": ["log"], "alpha": [0.0000001, 0.0000003, 0.000001, 0.000003, 0.00001, 0.00003,
#                                                  0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03,
#                                                  0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
#                       "l1_ratio": [0.2, 0.8]}
# _, sgd_gs_results_log = grid_search_sgd(param_grid_sgd_log, "log", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.8
#
# param_grid_sgd_log = {"loss": ["log"], "alpha": [0.0005, 0.001, 0.002], "l1_ratio": [0.6, 0.8, 0.9]}
# _, sgd_gs_results_log = grid_search_sgd(param_grid_sgd_log, "log_2", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.9
sgd_clf_log = SGDClassifier(random_state=8, penalty='elasticnet', alpha=0.001, l1_ratio=0.9)
# Precision = 0.58, Recall = 0.38, F1 score = 0.46


# # Grid search for squared hinge loss
# param_grid_sgd_squared_hinge = {"loss": ["squared_hinge"], "alpha": [0.0000001, 0.0000003, 0.000001, 0.000003,
#                                                                      0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003,
#                                                                      0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
#                                 "l1_ratio": [0.2, 0.8]}
# _, sgd_gs_results_squared_hinge = grid_search_sgd(param_grid_sgd_squared_hinge, "squared_hinge", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.8
#
# param_grid_sgd_squared_hinge = {"loss": ["squared_hinge"], "alpha": [0.0005, 0.001, 0.002], "l1_ratio": [0.6, 0.8, 0.9]}
# _, sgd_gs_results_squared_hinge = grid_search_sgd(param_grid_sgd_squared_hinge, "squared_hinge_2", 2, X_train_scaled, y_train) # Best: alpha = 0.0005, l1_ratio = 0.9
sgd_clf_squared_hinge = SGDClassifier(random_state=8, penalty='elasticnet', alpha=0.0005, l1_ratio=0.9)
# Precision = 0.56, Recall = 0.40, F1 score = 0.41


# # Grid search for perceptron loss
# param_grid_sgd_perceptron = {"loss": ["perceptron"], "alpha": [0.0000001, 0.0000003, 0.000001, 0.000003,
#                                                                0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003,
#                                                                0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
#                              "l1_ratio": [0.2, 0.8]}
# _, sgd_gs_results_perceptron= grid_search_sgd(param_grid_sgd_perceptron, "perceptron", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.8
#
# param_grid_sgd_perceptron = {"loss": ["perceptron"], "alpha": [0.0005, 0.001, 0.002], "l1_ratio": [0.6, 0.8, 0.9]}
# _, sgd_gs_results_perceptron = grid_search_sgd(param_grid_sgd_perceptron, "perceptron_2", 2, X_train_scaled, y_train) # Best: alpha = 0.0005, l1_ratio = 0.9
sgd_clf_perceptron = SGDClassifier(random_state=8, penalty='elasticnet', alpha=0.0005, l1_ratio=0.9)
# Precision = 0.56, Recall = 0.38, F1 score = 0.46


# # Grid search for modified_huber loss
# param_grid_sgd_modified_huber = {"loss": ["modified_huber"], "alpha": [0.0000001, 0.0000003, 0.000001, 0.000003,
#                                                                        0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003,
#                                                                        0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
#                                  "l1_ratio": [0.2, 0.8]}
# _, sgd_gs_results_modified_huber= grid_search_sgd(param_grid_sgd_modified_huber, "modified_huber", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.8
#
# param_grid_sgd_modified_huber = {"loss": ["modified_huber"], "alpha": [0.0005, 0.001, 0.002], "l1_ratio": [0.6, 0.8, 0.9]}
# _, sgd_gs_results_modified_huber = grid_search_sgd(param_grid_sgd_modified_huber, "modified_huber_2", 2, X_train_scaled, y_train) # Best: alpha = 0.001, l1_ratio = 0.9
sgd_clf_modified_huber = SGDClassifier(random_state=8, penalty='elasticnet', alpha=0.0001, l1_ratio=0.9)
# Precision = 0.58, Recall = 0.39, F1 score = 0.46

# Tuned SGD classifiers improved from default SGD around: Precision: 42%, Recall: 24%, F1 score: 32%


# Predict reduced y_scores and plot precision-recall curve for SGD classifiers

# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_hinge, X_train_scaled, y_train)
# plot_precision_recall_curve(y_true, y_scores_sgd, "Hinge")
#
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_log, X_train_scaled, y_train)
# plot_precision_recall_curve(y_true, y_scores_sgd, "Log")
#
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_squared_hinge, X_train_scaled, y_train)
# plot_precision_recall_curve(y_true, y_scores_sgd, "Squared hinge")
#
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_perceptron, X_train_scaled, y_train)
# plot_precision_recall_curve(y_true, y_scores_sgd, "Perceptron")
#
# y_true, y_scores_sgd = predict_scores_sgd(sgd_clf_modified_huber, X_train_scaled, y_train)
# plot_precision_recall_curve(y_true, y_scores_sgd, "Modified huber")

# Very similar behaviour. Not good precision/recall


# # Grid search for hinge loss (SVC) with robust scaling
# param_grid_sgd_hinge_robust = {"loss": ["hinge"], "alpha": [0.0000001, 0.0000003, 0.000001, 0.000003, 0.00001, 0.00003,
#                                                      0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03,
#                                                      0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
#                         "l1_ratio": [0.2, 0.8]}
# _, sgd_gs_results_hinge_robust = grid_search_sgd(param_grid_sgd_hinge_robust, "hinge_robust", 2, X_train_scaled_R, y_train) # Best: alpha = 0.000003, l1_ratio = 0.8
# Worse than standard scaling. Precision = 0.38, Recall = 0.31, F1 score = 0.34


# Plot epochs learning curves of SGD classifier to see if it converges
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def plot_SGD_epochs_learning_curve(max_epochs=1000, max_iter_sgd=100):
    if max_iter_sgd > max_epochs:
        print("max_epochs must be greater or equal to max_iter_sgd")
    else:
        # Split reduced data into train and validation sets
        X_train_learn, X_val_learn, y_train_learn, y_val_learn = train_test_split(X_train_scaled_red, y_train_red,
                                                                                  test_size=0.2, random_state=8,
                                                                                  stratify=y_train_red)

        # Classifier with fixed max_iter to check F1 score over epochs
        sgd_clf_step = SGDClassifier(max_iter=max_iter_sgd, random_state=8, penalty='elasticnet', loss="hinge",
                                     alpha=0.001, l1_ratio=0.9, warm_start=True, tol=-np.infty)

        # Create arrays of F1 scores over epochs
        n_epochs = int(max_epochs/max_iter_sgd)
        train_scores_array = np.empty(n_epochs)
        val_scores_array = np.empty(n_epochs)

        for epoch in range(n_epochs):
            # print(int((epoch+1) * max_iter_sgd))
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
        plt.figure()
        plt.plot(val_scores_array, "b-", linewidth=3, label="Validation set")
        plt.plot(train_scores_array, "r--", linewidth=2, label="Training set")
        plt.legend(loc="upper right", fontsize=14)
        plt.xlabel(("Epoch x" + str(max_iter_sgd)), fontsize=14)
        plt.ylabel("F1 score", fontsize=14)
        plt.show()

# plot_SGD_learning_curve(100, 10) # After around 40 epochs, the SGD classifier converges. No need to tune learning rate


# Plot class learning curves
from sklearn.model_selection import learning_curve

def plot_learning_curve(clf, X, y, train_sizes_n=10, scoring="f1", name=None):
    train_sizes = np.linspace(.1, 1.0, train_sizes_n)
    train_sizes, train_scores, val_scores = learning_curve(clf, X, y, cv=2, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Train score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="b", label="Validation score")
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    if name:
        plt.title(name)
    plt.show()


# SGD hinge learning curve for F1 score
# plot_learning_curve(sgd_clf_hinge, X_train_scaled_red, y_train_red, train_sizes_n=10, name="Reduced set - SGD hinge") # High bias


# High bias possible solution: Expand features with polynomial ones. Memory problem due sample size, oprion PCA reduction
from sklearn.preprocessing import PolynomialFeatures

# Expand X features with PolynomialFeatures
def poly_expand(X_train=X_train_red, X_test=X_test_red, degree=2, scale=True):
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X_train)
    X_train_poly = poly.transform(X_train)
    X_test_poly = poly.transform(X_test)
    if scale:
        X_train_poly, X_test_poly = scale_data(X_train_poly, X_test_poly)
    return X_train_poly, X_test_poly


# Dimension reduction using PCA
from sklearn.decomposition import PCA

# Returns X_train and X_test with n_components features after PCA
def pca_X(X_train=X_train, X_test=X_test,
          n_components=150, whiten=False, scale=True):
    pca = PCA(n_components, whiten=whiten)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    if scale:
        X_train_pca, X_test_pca = scale_data(X_train_pca, X_test_pca)
    return X_train_pca, X_test_pca


# # Test SGD classifiers without whiten
# X_train_pca, X_test_pca = pca_X()
# clf_scores(sgd_clf, X_train_pca, y_train, "SGD with PCA") # Worse recall
# clf_scores(sgd_clf_hinge, X_train_pca, y_train, "Tuned SGD with PCA (150 components)") # Worse recall
#
#
# # Test SGD classifiers with whiten
# X_train_pca, X_test_pca = pca_X(whiten=True)
# clf_scores(sgd_clf, X_train_pca, y_train, "SGD with PCA (Whiten)") # Worse precision, slight improvement the other against no whiten
# clf_scores(sgd_clf_hinge, X_train_pca, y_train, "Tuned SGD with PCA (Whiten)") # Worse precision, slight improvement the other against no whiten
#
#
# # Test SGD classifiers with whiten and more components
# X_train_pca, X_test_pca = pca_X(whiten=True, n_components=150)
# X_train_pca_scaled, _ = scale_data(X_train_pca, X_test_pca)
# clf_scores(sgd_clf_hinge, X_train_pca, y_train, "Tuned SGD with PCA (Whiten) and 150 components") # Almost equal to Tuned SGD w/ robust scaling
#
#
# # Test SGD classifiers with whiten and most components
# X_train_pca, X_test_pca = pca_X(whiten=True, n_components=190)
# X_train_pca_scaled, _ = scale_data(X_train_pca, X_test_pca)
# clf_scores(sgd_clf_hinge, X_train_pca, y_train, "Tuned SGD with PCA (Whiten) and 190 components") # Almost equal to Tuned SGD w/ robust scaling
# # Tuned SGD with PCA (Whiten) and 190 components
# # Train Precision = 0.5264
# # Test Precision = 0.5280
# # Train Recall = 0.4131
# # Test Recall = 0.4135
# # Train F1 score = 0.4627
# # Test F1 score = 0.4636


# Combine PCA with Polynomial features. Test on reduced sets
# X_train_pca, X_test_pca = pca_X(X_train_red, X_test_red, n_components=50, whiten=True)
# X_train_poly, X_test_poly = poly_expand(X_train_pca, X_test_pca)

# clf_scores(sgd_clf, X_train_poly, y_train_red, "PCA - Poly SGD")
# clf_scores(sgd_clf_hinge, X_train_poly, y_train_red, "PCA - Poly SGD hinge") # Promising but memory problem with big sample size data
# PCA - Poly SGD hinge
# Train Precision = 0.4787
# Test Precision = 0.2670
# Train Recall = 0.1308
# Test Recall = 0.0813
# Train F1 score = 0.2054
# Test F1 score = 0.1247


# # Test forest classifiers with whiten and most components
# X_train_pca, X_test_pca = pca_X(whiten=True, n_components=190)
# X_train_pca_scaled, _ = scale_data(X_train_pca, X_test_pca)
# clf_scores(forest_clf, X_train_pca, y_train, "Forest with PCA (Whiten) and 190 components") # Recall = 0


# Kernel approximatio
from sklearn.kernel_approximation import RBFSampler, Nystroem

def rbf_map(X_train=X_train_red, X_test=X_test_red, gamma=0.2,
           rbfsampler=True, n_components=100, scale=False):
    if rbfsampler:
        feature_map = RBFSampler(gamma=gamma, random_state=8,
                                 n_components=n_components)
    else:
        feature_map = Nystroem(gamma=gamma, random_state=8,
                               n_components=n_components)
    X_train_mapped = feature_map.fit_transform(X_train)
    X_test_mapped = feature_map.transform(X_test)
    if scale:
        X_train_mapped, X_test_mapped = scale_data(X_train_mapped, X_test_mapped)
    return X_train_mapped, X_test_mapped

# Create RBF kernel map of X and scale it
# X_train_rbf, X_test_rbf = rbf_map(X_train, X_test, n_components=100,
#                                   rbfsampler=False, scale=True)

# Test SGD classifier with RBF mapped features
# clf_scores(sgd_clf, X_train_rbf, y_train, "RBF kernel and default SGD") # High variance
# clf_scores(sgd_clf_hinge, X_train_rbf, y_train, "RBF kernel and SGD hinge") # High variance
# RBF kernel and SGD hinge
# Train Precision = 1.0000
# Test Precision = 0.0000
# Train Recall = 0.0007
# Test Recall = 0.0000
# Train F1 score = 0.0013
# Test F1 score = 0.0000




# Tuning forest hyper-parameters
# plot_learning_curve(forest_clf, X_train_scaled, y_train, train_sizes_n=5, name="Reduced set - Random forest", scoring="precision"), # High variance


# Grid search function for Random forest classifier
def grid_search_forest(param_grid, name, cv=2, X=X_train_scaled_red, y=y_train_red):
    forest_clf = RandomForestClassifier(random_state=8)
    scoring = ["precision", "recall", "f1"]
    df_path = "./GridSearch dataframes/forest_" + name + ".csv"
    return hyper_parameter_tuning(forest_clf, param_grid, scoring, df_path, cv=cv, refit_parameter="precision", X=X, y=y)


# # Grid search for n_estimators
# param_grid_forest_trees = {"n_estimators": [3, 10, 30, 100]}
# _, gs_results_forest_trees = grid_search_forest(param_grid_forest_trees, "trees", 2, X_train_scaled_red, y_train_red) # Best: n_estimators 3, 10
#
# param_grid_forest_trees = {"n_estimators": [5, 7]}
# _, gs_results_forest_trees = grid_search_forest(param_grid_forest_trees, "trees_2", 2, X_train_scaled_red, y_train_red) # Best:
#
# # Grid search for max_depth
# param_grid_forest_max_depth = {"max_depth": [3, 10, 30, None]}
# _, gs_results_forest_max_depth = grid_search_forest(param_grid_forest_max_depth, "max_depth", 2, X_train_scaled_red, y_train_red) # Best: max_depth 30
#
# param_grid_forest_max_depth = {"max_depth": [15, 60, 70, 80]} # Best: 40, 50
# _, gs_results_forest_max_depth = grid_search_forest(param_grid_forest_max_depth, "max_depth_3", 2, X_train_scaled_red, y_train_red) # Best: max_depth 30
#
# # Grid search for max_features
# param_grid_forest_max_features = {"max_features": [5, 15, 50, 100, 200]}
# _, gs_results_forest_max_features = grid_search_forest(param_grid_forest_max_features, "max_features", 2, X_train_scaled_red, y_train_red) # Too much memory

# Grid search for max_leaf_nodes
# param_grid_forest_max_leaf_nodes = {"max_leaf_nodes": [125, 140, 160, 180, 210]}
# _, gs_results_forest_max_leaf_nodes = grid_search_forest(param_grid_forest_max_leaf_nodes, "max_leaf_nodes_2", 2, X_train_red, y_train_red) # Best: 150 0.75 Precision - 210 0.77 Precision
#
# # Grid search for
# param_grid_forest_ = {"": []}
# _, gs_results_forest_ = grid_search_forest(param_grid_forest_, "", 2, X_train_scaled_red, y_train_red) # Best:



# forest_clf_tuned = RandomForestClassifier(random_state=8, n_estimators=5, max_depth=35, max_leaf_nodes=210)
# clf_scores(forest_clf_tuned, X_train_red, y_train_red, title="Tuned forest")
# # Tuned forest
# # Train Precision = 0.9916
# # Test Precision = 0.4644
# # Train Recall = 0.1031
# # Test Recall = 0.0077
# # Train F1 score = 0.1868
# # Test F1 score = 0.0152


# PCA reduction and testing with forest
# X_train_pca_1, X_test_pca_1 = pca_X(X_train_red, X_test_red, n_components=150, whiten=False, scale=False)
#
# X_train_pca_2, X_test_pca_2 = pca_X(X_train_red, X_test_red, n_components=150, whiten=True, scale=False)
#
# X_train_pca_3, X_test_pca_3 = pca_X(X_train_red, X_test_red, n_components=100, whiten=False, scale=False)
#
# X_train_pca_4, X_test_pca_4 = pca_X(X_train_red, X_test_red, n_components=100, whiten=True, scale=False)
#
#
# clf_scores(forest_clf, X_train_pca_1, y_train_red, title="Random forest with pca_1") # Test Precision = 0.4520
# clf_scores(forest_clf, X_train_pca_2, y_train_red, title="Random forest with pca_2") # Test Precision = 0.4651
# clf_scores(forest_clf, X_train_pca_3, y_train_red, title="Random forest with pca_3") # The other two very bad for the 4 cases
# clf_scores(forest_clf, X_train_pca_4, y_train_red, title="Random forest with pca_4")
# Random forest with pca_2
# Train Precision = 0.9999
# Test Precision = 0.4651
# Train Recall = 0.8447
# Test Recall = 0.0070
# Train F1 score = 0.9157
# Test F1 score = 0.0137


# forest_clf_best, _ = grid_search_forest_max_features()
# Best max_features = 'auto'
# forest_clf_best, _ = grid_search_forest_n_estimators()
# Best n_estimators = 100
# print_precision_recall_f1_score(forest_clf_best)
# Precision = 0.9659
# Recall = 0.9658
# F1 score = 0.9658








# Test classifiers on X_train and X_test

from sklearn.base import clone

# Scores of classifier in train and test sets
def test_scoring(clf, X_train=X_train_scaled, X_test=X_test_scaled,
                 y_train=y_train, y_test=y_test, name=None):
    if name:
        print(name)

    clf_test = clone(clf)
    clf_test.fit(X_train, y_train)
    # clf_test.fit(X_train_scaled, y_train.ravel())

    y_train_predict = clf_test.predict(X_train)
    y_test_predict = clf_test.predict(X_test)

    train_precision = precision_score(y_train, y_train_predict)
    test_precision = precision_score(y_test, y_test_predict)
    print("Train Precision: ", train_precision)
    print("Test Precision: ", test_precision)

    train_recall = recall_score(y_train, y_train_predict)
    test_recall = recall_score(y_test, y_test_predict)
    print("Train Recall: ", train_recall)
    print("Test Recall: ", test_recall)

    train_f1 = f1_score(y_train, y_train_predict)
    test_f1 = f1_score(y_test, y_test_predict)
    print("Train F1: ", train_f1)
    print("Test F1: ", test_f1)

    return [test_precision, test_recall, test_f1, train_precision, train_recall, train_f1]


# Bar plot of train and test scores
def scores_bar(scores, title=None):
    labels = ["Precision", "Recall", "F1 score"]
    train_scores = scores[3:]
    test_scores = scores[:3]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, test_scores, width, label='Test')
    rects2 = ax.bar(x + width / 2, train_scores, width, label='Train')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.ylim([0, 1])
    # fig.tight_layout()
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Scores')
    plt.show()

# sgd_test_scores = test_scoring(sgd_clf_hinge, X_train_scaled, X_test_scaled, y_train, y_test, "Tuned SGD")
# scores_bar(sgd_test_scores, "SGD")
# Tuned SGD
# Train Precision:  0.6017008504252126
# Test Precision:  0.588871096877502
# Train Recall:  0.37405149894265455
# Test Recall:  0.36592039800995024
# Train F1:  0.4613201396080236
# Test F1:  0.451365449524394


# Confussion matrix of classifiers
from sklearn.metrics import confusion_matrix

def conf_matrix(clf, X_train=X_train_scaled, X_test=X_test_scaled,
                 y_train=y_train, y_test=y_test):
    clf_test = clone(clf)
    clf_test.fit(X_train, y_train)
    # clf_test.fit(X_train_scaled, y_train.ravel())

    y_train_predict = clf_test.predict(X_train)
    y_test_predict = clf_test.predict(X_test)

    test_matrix = confusion_matrix(y_test, y_test_predict)
    train_matrix = confusion_matrix(y_train, y_train_predict)
    return test_matrix

print(conf_matrix(sgd_clf_hinge))
# [[34953  1027]
#  [ 2549  1471]]
# Recall is very low, a lot of positive transactions are not predicted (2549 from a total of 4020)
# depending on what should be more important, the precision/recall tradeoff could be adjusted

print(conf_matrix(forest_clf))
# [[35908    72]
#  [ 3935    85]]


# High bias of SGD not solved. Nonetheless it performs as a classifier
# High variance of forest not solved. Model can not be said to be a classifier due its terrible performance on the test set