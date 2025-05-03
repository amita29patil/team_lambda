import os
import joblib

def load_best_models(models_folder):
    """
    Load best models from the specified folder.

    Args:
        models_folder (str): Path to the folder containing the model files.

    Returns:
        dict: A dictionary of loaded models with their names as keys.
    """
    models = {
        "xgb_model": joblib.load(os.path.join(models_folder, 'best_xgb_model.pkl')),
        "rf_model": joblib.load(os.path.join(models_folder, 'best_rf_model.pkl')),
        "log_reg_model": joblib.load(os.path.join(models_folder, 'best_log_reg_model.pkl')),
        "nn_model": joblib.load(os.path.join(models_folder, 'best_nn_model.pkl'))
    }
    print("Best Models successfully loaded from:", models_folder)
    return models


def load_pre_tuned_models(models_folder):
    """
    Load pre-tuned models from the specified folder.

    Args:
        models_folder (str): Path to the folder containing the model files.

    Returns:
        dict: A dictionary of loaded models with their names as keys.
    """
    models = {
        "xgb_pre_tuned": joblib.load(os.path.join(models_folder, 'xgboost_pre_tuned.pkl')),
        "rf_pre_tuned": joblib.load(os.path.join(models_folder, 'random_forest_pre_tuned.pkl')),
        "log_reg_pre_tuned": joblib.load(os.path.join(models_folder, 'logistic_regression_pre_tuned.pkl')),
        "nn_model_pre_tuned": joblib.load(os.path.join(models_folder, 'neural_network_pre_tuned.pkl'))
    }
    print("Pre-tuned models successfully loaded from:", models_folder)
    return models