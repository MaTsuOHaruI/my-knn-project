import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------
# Logistic Regression Class Definition
# ------------------------
class LogisticModel:
    def __init__(self, max_iter=100, solver='liblinear'):
        # Initialize logistic regression model with specified solver and max iterations
        self.model = LogisticRegression(max_iter=max_iter, solver=solver)
    
    def fit(self, X_train, y_train):
        # Fit the model using training data
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        # Predict using trained model
        return self.model.predict(X)
    
    def accuracy(self, y_true, y_pred):
        # Calculate accuracy
        return np.mean(y_true == y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        # Return confusion matrix
        return confusion_matrix(y_true, y_pred)


# ------------------------
# Data Preprocessing Functions
# ------------------------

def standardize(X_train, X_test):
    # Standardize features using training data's mean and standard deviation
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def remove_outliers(X_train, y_train):
    # Remove outliers using the IQR method for each column
    for column in X_train.columns:
        q1 = np.percentile(X_train[column], 25, method='midpoint')
        q3 = np.percentile(X_train[column], 75, method='midpoint')
        iqr = q3 - q1
        drop_indices = X_train[(X_train[column] >= q3 + 1.5 * iqr) | (X_train[column] <= q1 - 1.5 * iqr)].index
        X_train = X_train.drop(drop_indices)
        y_train = y_train.drop(drop_indices)
    return X_train, y_train

def fill_zero(X_train, X_test):
    # Replace zero values in each column with the column's mean
    for column in X_train.columns:
        avg = X_train[column].mean()
        X_train[column] = X_train[column].replace(0, avg)
        X_test[column] = X_test[column].replace(0, avg)
    return X_train, X_test

def remove_low_corr(X_train, X_test, y_train, threshold=0.1):
    # Remove features that have low correlation with the target variable
    corr = X_train.corrwith(y_train)
    drop_cols = [col for col in X_train.columns if abs(corr[col]) < threshold]
    return X_train.drop(columns=drop_cols), X_test.drop(columns=drop_cols)

def balance_data(X_train, y_train):
    # Downsample majority class (assumed to be class 0) to balance the dataset
    zero_idx = y_train[y_train == 0].index
    drop_zero = np.random.choice(zero_idx, int(len(zero_idx) * 0.1), replace=False)
    return X_train.drop(drop_zero, errors='ignore'), y_train.drop(drop_zero, errors='ignore')


# ------------------------
# Evaluation Function
# ------------------------
def evaluate_step(step_name, X_train_step, X_test_step, y_train_step, y_test_step):
    print(f"Evaluating step: {step_name}")
    print(f"[{step_name}] Train size: {len(X_train_step)}, Test size: {len(X_test_step)}")

    # Reset indices to avoid mismatch issues
    X_train_step_reset = pd.DataFrame(X_train_step).reset_index(drop=True)
    X_test_step_reset = pd.DataFrame(X_test_step).reset_index(drop=True)
    y_train_step_reset = y_train_step.reset_index(drop=True)
    y_test_step_reset = y_test_step.reset_index(drop=True)

    # Initialize and train logistic regression model
    logistic_model = LogisticModel(max_iter=100)
    logistic_model.fit(X_train_step_reset, y_train_step_reset)
    
    # Make predictions on test data
    y_pred = logistic_model.predict(X_test_step_reset)
    
    # Print accuracy and confusion matrix
    accuracy = logistic_model.accuracy(y_test_step_reset, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("Confusion Matrix:")
    print(logistic_model.confusion_matrix(y_test_step_reset, y_pred))


# ------------------------
# Main Process
# ------------------------

# 1. Load dataset
data = pd.read_csv('train_data_A.csv')
print(f"Total data size: {len(data)}")  # Print total dataset size

# 2. Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3. Shuffle and split data into training and testing sets (70/30 split)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train_raw, X_test_raw = X.iloc[train_indices], X.iloc[test_indices]
y_train_raw, y_test = y.iloc[train_indices], y.iloc[test_indices]

# --- Original Data ---
evaluate_step("Original", X_train_raw.copy(), X_test_raw.copy(), y_train_raw.copy(), y_test)

# --- Standardized Data ---
X_train_std, X_test_std = standardize(X_train_raw.copy(), X_test_raw.copy())
evaluate_step("Standardized", X_train_std, X_test_std, y_train_raw.copy(), y_test)

# --- Outliers Removed ---
X_train_out, y_train_out = remove_outliers(X_train_raw.copy(), y_train_raw.copy())
evaluate_step("Removed Outliers", X_train_out.copy(), X_test_raw.copy(), y_train_out.copy(), y_test)

# --- Zero Replaced ---
print("Zero count (Before):", (X_train_raw == 0).sum())  # Before filling zeroes
X_train_fill, X_test_fill = fill_zero(X_train_raw.copy(), X_test_raw.copy())
print("Zero count (After Fill):", (X_train_fill == 0).sum())  # After filling zeroes
evaluate_step("Filled 0", X_train_fill.copy(), X_test_fill.copy(), y_train_raw.copy(), y_test)

# --- Low-Correlation Features Removed ---
X_train_corr, X_test_corr = fill_zero(X_train_raw.copy(), X_test_raw.copy())  # Fill zero first
X_train_corr, X_test_corr = remove_low_corr(X_train_corr, X_test_corr, y_train_raw.copy())  # Then remove low-corr
evaluate_step("Removed Low-Corr", X_train_corr.copy(), X_test_corr.copy(), y_train_raw.copy(), y_test)

# --- Balanced Data ---
X_train_bal, X_test_bal = fill_zero(X_train_raw.copy(), X_test_raw.copy())  # Fill zero
X_train_bal, X_test_bal = remove_low_corr(X_train_bal, X_test_bal, y_train_raw.copy())  # Remove low-corr
X_train_bal, y_train_bal = balance_data(X_train_bal, y_train_raw.copy())  # Finally balance the dataset
evaluate_step("Balanced", X_train_bal.copy(), X_test_bal.copy(), y_train_bal.copy(), y_test)
