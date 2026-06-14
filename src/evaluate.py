# src/evaluate.py
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test, feature_names, config):
    """Generates evaluation metrics, confusion matrix, ROC curve, and Feature Importance."""
    output_dir = config['paths']['outputs']
    y_pred = model.predict(X_test)

    # 1. Classification Report
    logger.info("\n" + classification_report(y_test, y_pred))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Diseased'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close(fig)

    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close(fig)
    logger.info(f"Evaluation plots saved to {output_dir}")

    # 4. Feature Importance (Extracting weights from Logistic Regression)
    logreg_model = model.named_steps['logreg']
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': logreg_model.coef_[0]
    }).sort_values(by='Importance', key=abs, ascending=False)
    
    logger.info("\nTop 5 Most Important Features:\n" + importance.head().to_string(index=False))