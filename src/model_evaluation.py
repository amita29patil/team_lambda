from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nClassification Report for {model_name}:\n", classification_report(y_test, y_pred))
    print(f"ROC AUC Score for {model_name}: {roc_auc_score(y_test, y_proba)}")

    # Plot ROC Curve
    plot_roc_curve(y_test, y_proba, model_name)

    # Plot Precision-Recall Curve
    plot_precision_recall_curve(y_test, y_proba, model_name)

def plot_roc_curve(y_test, y_proba, model_name):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_test, y_proba, model_name):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="teal", lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()