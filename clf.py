from data_processing import convert_to_df, bin_frames, vectorize
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

random_state = 5
matrix_save_dir = "saved_matrices"

file_paths = {
    "cla1-1.pcap": 1, "cla2-1.pcap" : 2, "cla3-1.pcap": 3}

def matrix_to_capture_features(X_matrix: np.ndarray) -> np.ndarray:
    """
    Convert one capture matrix (n_bins, 9) into one fixed-length feature vector.
    """
    if X_matrix.size == 0:
        return np.zeros((37,), dtype=float)

    feature_vector = np.concatenate([
        X_matrix.mean(axis=0),
        X_matrix.std(axis=0),
        X_matrix.min(axis=0),
        X_matrix.max(axis=0),
        np.array([X_matrix.shape[0]], dtype=float),
    ])
    return feature_vector

X_capture = []
y_capture = []
for path, label in file_paths.items():
    df = convert_to_df(path)
    
    df = bin_frames(df)
    df["label"] = label
    X_matrix, y = vectorize(
        df,
        save_dir=matrix_save_dir,
        matrix_name=Path(path).stem,
    )
    X_capture.append(matrix_to_capture_features(X_matrix))
    y_capture.append(int(y[0]))

X_cumulative = np.vstack(X_capture).astype(float)
y_cumulative = np.array(y_capture, dtype=int)
    
print("Feature matrix X shape:", X_cumulative.shape)
print("Label vector y shape:", y_cumulative.shape)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X_train, X_val, y_train, y_val = train_test_split(X_cumulative, y_cumulative, test_size=0.15, random_state=5)
    
# Define the scaler for scaling the data
scaler = StandardScaler()

# Fit the scaler to the training data
scaler.fit(X_train)

# Normalize the training data
X_train = scaler.transform(X_train)

# Use the scaler defined above to standardize the validation data by applying the same transformation to the validation data.
X_val = scaler.transform(X_val)

rbf_clf = SVC(kernel='rbf', degree=3, random_state=random_state)
rbf_clf.fit(X_train, y_train)

poly_clf = SVC(kernel='poly', degree=3, random_state=random_state)
poly_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_train_pred_rbf = rbf_clf.predict(X_train)
print("RBF Kernel SVM Classification Report On Training Set:")
print(classification_report(y_train, y_train_pred_rbf))

y_test_pred_rbf = rbf_clf.predict(X_val)
print("RBF Kernel SVM Classification Report On Validation Set:")
print(classification_report(y_val, y_test_pred_rbf))

y_train_pred_poly = poly_clf.predict(X_train)
print("Polynomial Kernel SVM Classification Report:")
print(classification_report(y_train, y_train_pred_poly))

y_test_pred_poly = poly_clf.predict(X_val)
print("Polynomial Kernel SVM Classification Report On Validation Set:")
print(classification_report(y_val, y_test_pred_poly))






    
