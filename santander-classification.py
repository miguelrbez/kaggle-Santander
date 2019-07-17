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


# Split data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=8, stratify=y)

# Check whether the split has equitative positive and negative labels
# print((y_train == True).sum() / len(y_train)) # 0.1004875
# print((y_test == True).sum() / len(y_test)) # 0.1005


# Histogram of features mean values
# plt.figure()
# plt.hist(features_mean_values)
# plt.show()

# The features mean values has a Gaussian shape, maybe some features are correlated


# Histogram of features std values
# plt.figure()
# plt.hist(features_std_values)
# plt.show()

# The features std values are mostly close to 0, this could lead to the correlation hypothesis


# Correlation matrix of features
# corr_matrix = pd.DataFrame(X_train).corr().values

# Mean correlation value of features
# mean_corr_values = np.mean(corr_matrix, axis=0)


# Scale features
from sklearn.preprocessing import StandardScaler

def scale_data():
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

X_train_scaled, X_test_scaled = scale_data()
