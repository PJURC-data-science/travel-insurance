from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_curve, roc_auc_score

def chi_squared_test(df: pd.DataFrame, var_1: str, var_2: str) -> None:
    """
    Performs a chi-squared test on the given DataFrame and returns the p-value.

    Args:
        df (DataFrame): Input DataFrame
        var_1 (str): Name of the variable to perform the chi-squared test on
        var_2 (str): Name of the variable to perform the chi-squared test on

    Returns:
        None
    """
    
    contingency_table = pd.crosstab(df[var_1], df[var_2])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")


def visualize_performance(y_test: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series) -> None:
    """
    Visualizes the model performance

    Args:
        y_test (ndarray): True test labels
        y_pred (ndarray): Predicted labels
        y_pred_proba (ndarray): Predicted probabilities

    Returns:
        None
    """
    # 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    ax[0].set_title('Confusion Matrix')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax[1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')
    ax[1].legend(loc='lower right')

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax[2].plot(recall, precision)
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_title('Precision-Recall Curve')
    
    plt.tight_layout()
    plt.show()