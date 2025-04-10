import numpy as np
import pandas as pd

# ------------------------
# KNN Class Definition
# ------------------------
class Knn:
    def __init__(self, k=3, dist_func=None, weight=None):
        self.k = k
        self.dist_func = dist_func if dist_func else lambda x, y: np.linalg.norm(x - y)  # Default to Euclidean distance
        self.weight = weight  # 'isd' means inverse squared distance
        self.X_train = np.array([])
        self.y_train = np.array([])

    def fit(self, X_train, y_train):
        # Convert to NumPy array if DataFrame/Series
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.values if isinstance(y_train, pd.Series) else y_train

    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        preds = np.array([])

        for x in X:
            # Calculate distances from test point x to all training points
            distances = np.array([self.dist_func(x, p) for p in self.X_train])
            sorted_indices = np.argsort(distances)
            distances = distances[sorted_indices]
            sorted_y = self.y_train[sorted_indices]

            if not self.weight:
                # Regular voting
                pred = np.bincount(sorted_y[:self.k]).argmax()
            elif self.weight in ['isd', 'inverse_squared_distance']:
                # Weighted voting based on inverse squared distance
                probs = []
                classes = np.unique(self.y_train)
                for i in classes:
                    class_indices = np.where(sorted_y[:self.k] == i)
                    prob = np.sum(1 / distances[class_indices]**2) if len(class_indices[0]) > 0 else 0
                    probs.append(prob)
                pred = classes[np.argmax(probs)]
            else:
                raise AttributeError('Weight name not found')

            preds = np.append(preds,pred)

        return preds

    def accuracy(self, y_true, y_pred):
        # Compute accuracy
        return np.mean(y_true == y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        # Compute confusion matrix [ [TN, FP], [FN, TP] ]
        TP = np.sum((y_true == 1) & (y_pred == 1)) #True Positives
        FP = np.sum((y_true == 0) & (y_pred == 1)) #False Positives
        TN = np.sum((y_true == 0) & (y_pred == 0)) #True Negatives
        FN = np.sum((y_true == 1) & (y_pred == 0)) #False Negatives

        return np.array([[TN, FP], [FN, TP]])
    



# ------------------------
# K tuning function
# ------------------------
def find_best_k(model: Knn, X_val, y_val, k_min: int, k_max: int, save_best_k: bool=True, odd_k_only=False):
    if len(model.X_train) == 0 or len(model.y_train) == 0:
        raise AttributeError("Model has no training data")

    model.k_scores = {'k': np.array([]), 'scores': np.array([])}
    for k in range(k_min, k_max + 1):
        if odd_k_only and k % 2 == 0: continue
        model.k = k
        y_pred = model.predict(X_val)
        score = model.accuracy(y_val, y_pred)
        model.k_scores['k'] = np.append(model.k_scores['k'],k)
        model.k_scores['scores'] = np.append(model.k_scores['score'], score)
        print(f"k={k}, Accuracy: {model.k_scores['score'][-1]}")
        print(model.confusion_matrix(y_val, y_pred))

    if save_best_k:
        best_k = int(model.k_scores['k'][np.argmax(model.k_scores['scores'])])
        model.k = best_k
        return model.k


# ------------------------
# Preprocessing functions
# ------------------------

def standardize(X_train, X_test):
    # Standardize features using mean and std of train
    return (X_train - X_train.mean()) / X_train.std(), (X_test - X_test.mean()) / X_test.std()

def remove_outliers(X_train, y_train):
    # Remove outliers using IQR method
    for column in X_train.columns:
        q1 = np.percentile(X_train[column], 25, method='midpoint')
        q3 = np.percentile(X_train[column], 75, method='midpoint')
        iqr = q3 - q1
        drop_indices = X_train[(X_train[column] >= q3 + 1.5 * iqr) | (X_train[column] <= q1 - 1.5 * iqr)].index
        X_train = X_train.drop(drop_indices)
        y_train = y_train.drop(drop_indices)
    return X_train, y_train

def fill_zero(X_train, X_test):
    # Replace zeroes with column average
    for column in X_train.columns:
        avg = X_train[column].mean()
        X_train[column] = X_train[column].replace(0, avg)
        X_test[column] = X_test[column].replace(0, avg)
    return X_train, X_test

def remove_low_corr(X_train, X_test, y_train, threshold=0.1):
    # Remove features with low correlation to target
    corr = X_train.corrwith(y_train)
    drop_cols = [col for col in X_train.columns if abs(corr[col]) < threshold]
    return X_train.drop(columns=drop_cols), X_test.drop(columns=drop_cols)


def balance_data(X_train, y_train):
    # Downsample majority class (class 0)
    zero_idx = y_train[y_train == 0].index
    drop_zero = np.random.choice(zero_idx, int(len(zero_idx) * 0.1), replace=False)
    return X_train.drop(drop_zero, errors='ignore'), y_train.drop(drop_zero, errors='ignore')

# ------------------------
# Evaluation function
# ------------------------

def evaluate_step(step_name, X_train_step, X_test_step, y_train_step, y_test_step):
    print(f"Evaluating step: {step_name}")
    print(f"[{step_name}] Train size: {len(X_train_step)}, Test size: {len(X_test_step)}")
    print(f"[STEP] {step_name} - Train: {len(X_train_step)}, Test: {len(X_test_step)}")

    # Reset indices to avoid mismatch errors
    X_train_step_reset = X_train_step.reset_index(drop=True)
    X_test_step_reset = X_test_step.reset_index(drop=True)
    y_train_step_reset = y_train_step.reset_index(drop=True)
    y_test_step_reset = y_test_step.reset_index(drop=True)

    # Train the KNN model
    knn = Knn(k=21)
    knn.fit(X_train_step_reset, y_train_step_reset)
    
    # Make predictions
    y_pred = knn.predict(X_test_step_reset)
    
    # Print accuracy and confusion matrix
    accuracy = knn.accuracy(y_test_step_reset, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("Confusion Matrix:")
    print(knn.confusion_matrix(y_test_step_reset, y_pred))

    


# ------------------------
# Main Process
# ------------------------

# 1. Load & split
data = pd.read_csv(r'C:\Users\haru2\OneDrive\Desktop\Group15\train_data_A.csv')
print(f"Total data size: {len(data)}")  # Print data size

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Shuffle and split into 70% train and 30% test
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train_raw, X_test_raw = X.iloc[train_indices], X.iloc[test_indices]
y_train_raw, y_test = y.iloc[train_indices], y.iloc[test_indices]

# --- Original ---
evaluate_step("Original", X_train_raw.copy(), X_test_raw.copy(), y_train_raw.copy(), y_test)

# --- Standardized ---
X_train_std, X_test_std = standardize(X_train_raw.copy(), X_test_raw.copy())
evaluate_step("Standardized", X_train_std, X_test_std, y_train_raw.copy(), y_test)

# --- Outlier Removed ---
X_train_out, y_train_out = remove_outliers(X_train_raw.copy(), y_train_raw.copy())
evaluate_step("Removed Outliers", X_train_out.copy(), X_test_raw.copy(), y_train_out.copy(), y_test)

# --- Fill 0 ---
print("Zero count (Before):", (X_train_raw == 0).sum())  # Before missing values
X_train_fill, X_test_fill = fill_zero(X_train_raw.copy(), X_test_raw.copy())
print("Zero count (After Fill):", (X_train_fill == 0).sum())  # After missing values
evaluate_step("Filled 0", X_train_fill.copy(), X_test_fill.copy(), y_train_raw.copy(), y_test)

# --- Low-Corr Removed ---
X_train_corr, X_test_corr = fill_zero(X_train_raw.copy(), X_test_raw.copy())  # fill_zero first
X_train_corr, X_test_corr = remove_low_corr(X_train_corr, X_test_corr, y_train_raw.copy())
evaluate_step("Removed Low-Corr", X_train_corr.copy(), X_test_corr.copy(), y_train_raw.copy(), y_test)

# --- Balanced ---
X_train_bal, X_test_bal = fill_zero(X_train_raw.copy(), X_test_raw.copy())  # fill_zero first
X_train_bal, X_test_bal = remove_low_corr(X_train_bal, X_test_bal, y_train_raw.copy())  # low-corr
X_train_bal, y_train_bal = balance_data(X_train_bal, y_train_raw.copy())  #  Balancing at the end
evaluate_step("Balanced", X_train_bal.copy(), X_test_bal.copy(), y_train_bal.copy(), y_test)
