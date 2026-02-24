# decision_tree_training.py

import os
import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from synthetic_generator import generate_synthetic_data


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree_model.pkl")


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


# =============================================================================
# PREPROCESSOR
# =============================================================================

class DatasetProcessor:
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit_transform(self, x: pd.DataFrame, y: pd.Series):
        x_encoded = x.copy()

        for col in x.columns:
            le = LabelEncoder()
            x_encoded[col] = le.fit_transform(x[col])
            self.encoders[col] = le

        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.encoders["target"] = target_encoder

        return x_encoded, y_encoded


# =============================================================================
# MODEL TRAINER
# =============================================================================

class DecisionTreeTrainer:

    def __init__(self):
        self.model = None

    def tune_and_train(self, x_train, y_train):
        param_grid = {
            "max_depth": range(1, 15),
            "criterion": ["gini", "entropy"]
        }

        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            param_grid,
            cv=5,
            scoring="accuracy"
        )

        grid.fit(x_train, y_train)

        self.model = grid.best_estimator_

        logger.info(f"Best Parameters: {grid.best_params_}")
        return self.model


# =============================================================================
# EVALUATION
# =============================================================================

class Evaluator:

    @staticmethod
    def generate_all_graphs(model, x_test, y_test, feature_names):

        prediction = model.predict(x_test)
        probs = model.predict_proba(x_test)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, prediction)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=["No", "Yes"],
                    yticklabels=["No", "Yes"])
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
        plt.close()

        # Feature Importance
        plt.figure()
        plt.bar(feature_names, model.feature_importances_)
        plt.title("Feature Importance")
        plt.savefig(os.path.join(GRAPH_DIR, "feature_importance.png"))
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1])
        plt.legend()
        plt.title("ROC Curve")
        plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
        plt.close()

        # Precision Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision Recall Curve")
        plt.savefig(os.path.join(GRAPH_DIR, "precision_recall_curve.png"))
        plt.close()

        # Tree Structure
        plt.figure(figsize=(12, 6))
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=["No", "Yes"],
            filled=True
        )
        plt.title("Decision Tree Structure")
        plt.savefig(os.path.join(GRAPH_DIR, "tree_structure.png"))
        plt.close()


# =============================================================================
# PIPELINE
# =============================================================================

class MLPipeline:

    def run(self):

        logger.info("Generating synthetic dataset...")
        x, y = generate_synthetic_data(n_samples=1000, save=True)

        processor = DatasetProcessor()
        x, y = processor.fit_transform(x, y)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE
        )

        trainer = DecisionTreeTrainer()
        model = trainer.tune_and_train(x_train, y_train)

        prediction = model.predict(x_test)

        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_test, prediction),
            precision=precision_score(y_test, prediction),
            recall=recall_score(y_test, prediction),
            f1_score=f1_score(y_test, prediction)
        )

        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Precision: {metrics.precision:.4f}")
        logger.info(f"Recall: {metrics.recall:.4f}")
        logger.info(f"F1 Score: {metrics.f1_score:.4f}")

        Evaluator.generate_all_graphs(
            model,
            x_test,
            y_test,
            x.columns
        )

        joblib.dump({
            "model": model,
            "encoders": processor.encoders
        }, MODEL_PATH)

        logger.info(f"Model saved at: {MODEL_PATH}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    MLPipeline().run()


if __name__ == "__main__":
    main()
