import os
import joblib

def load_data(data_folder):
    """
    Load train and test datasets from the specified folder.

    Args:
        data_folder (str): Path to the folder containing the data files.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load the datasets
    X_train = joblib.load(os.path.join(data_folder, 'X_train.pkl'))
    X_test = joblib.load(os.path.join(data_folder, 'X_test.pkl'))
    y_train = joblib.load(os.path.join(data_folder, 'y_train.pkl'))
    y_test = joblib.load(os.path.join(data_folder, 'y_test.pkl'))

    print("Data successfully loaded from:", data_folder)
    return X_train, X_test, y_train, y_test