"""
SVM.py
Enterprise Structured Support Vector Machine Pipeline
Loan Approval Prediction System
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import warnings
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# 2. CONFIGURATION LAYER
# ============================================================

class Config:

    RANDOM_STATE = 42
    DATASET_PATH = "loan_data.csv"
    TARGET_COLUMN = "loan_status"

    TEST_SIZE = 0.2

    # SVM Default Parameters
    SVM_PARAMS = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "degree": 3,
        "probability": True,
        "random_state": RANDOM_STATE,
    }

    # Grid Search Parameters
    GRID_PARAMS = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    }

    CV_FOLDS = 5
    TUNE_FOLDS = 3

    MODEL_PATH = "SVM\models\svm_model.pkl"

    FIG_SIZE = (12, 8)
    DPI = 100
    STYLE = "seaborn-v0_8-darkgrid"

    LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


# Setup logging
logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ============================================================
# 3. DATA LAYER
# ============================================================

class DataLoader:
    """
    Responsible for loading dataset.
    Falls back to synthetic generation if CSV not found.
    """

    def __init__(self, path: str):
        self.path = path
        self.data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.feature_names: Optional[list] = None

    # --------------------------------------------------------
    # Synthetic Dataset Generator
    # --------------------------------------------------------

    def _generate_synthetic(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:

        logger.warning("Dataset not found. Generating synthetic loan dataset.")

        np.random.seed(Config.RANDOM_STATE)

        n_approved = n_samples // 2
        n_rejected = n_samples - n_approved

        def generate_block(n, approved: bool):

            income_mean = 70000 if approved else 35000
            credit_mean = 700 if approved else 550
            loan_mean = 8000 if approved else 20000
            rate_mean = 10 if approved else 15

            return pd.DataFrame({
                "person_age": np.random.randint(20, 70, n),
                "person_income": np.random.normal(income_mean, 20000, n).clip(10000, 200000),
                "credit_score": np.random.normal(credit_mean, 60, n).clip(300, 850),
                "loan_amnt": np.random.normal(loan_mean, 5000, n).clip(500, 35000),
                "loan_int_rate": np.random.normal(rate_mean, 3, n).clip(5, 25),
                "person_emp_exp": np.random.randint(0, 15, n),
                "person_gender": np.random.choice(["male", "female"], n),
                "person_home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n),
                "loan_intent": np.random.choice(
                    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"], n
                ),
                "previous_loan_defaults_on_file":
                    np.random.choice(["No", "Yes"], n, p=[0.8, 0.2] if approved else [0.4, 0.6]),
            })

        approved_df = generate_block(n_approved, True)
        rejected_df = generate_block(n_rejected, False)

        data = pd.concat([approved_df, rejected_df], ignore_index=True)
        target = pd.Series(
            np.concatenate([np.ones(n_approved), np.zeros(n_rejected)]),
            name=Config.TARGET_COLUMN,
        ).astype(int)

        idx = np.random.permutation(n_samples)

        return data.iloc[idx].reset_index(drop=True), target.iloc[idx].reset_index(drop=True)

    # --------------------------------------------------------
    # Main Loader
    # --------------------------------------------------------

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:

        logger.info("Loading dataset")

        if os.path.exists(self.path):
            df = pd.read_csv(self.path)

            if Config.TARGET_COLUMN not in df.columns:
                raise ValueError(f"Target column '{Config.TARGET_COLUMN}' missing")

            self.data = df.drop(Config.TARGET_COLUMN, axis=1)
            self.target = df[Config.TARGET_COLUMN]

            logger.info(f"Dataset loaded with {len(df)} samples")
        else:
            self.data, self.target = self._generate_synthetic()

        self.feature_names = list(self.data.columns)

        return self.data, self.target


# ------------------------------------------------------------
# Data Validator
# ------------------------------------------------------------

class DataValidator:
    """
    Responsible only for validating raw dataset.
    No transformations allowed.
    """

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target

    def validate(self) -> bool:

        logger.info("Validating dataset")

        if self.data.empty or self.target.empty:
            logger.error("Dataset is empty")
            return False

        if len(self.data) != len(self.target):
            logger.error("Feature and target size mismatch")
            return False

        missing_features = self.data.isnull().sum().sum()
        missing_target = self.target.isnull().sum()

        logger.info(f"Missing feature values: {missing_features}")
        logger.info(f"Missing target values: {missing_target}")

        inf_count = np.isinf(
            self.data.select_dtypes(include=[np.number]).values
        ).sum()

        logger.info(f"Infinite numeric values: {inf_count}")

        class_dist = self.target.value_counts()
        logger.info(f"Class distribution:\n{class_dist}")

        logger.info("Dataset validation completed")

        return True
    
# ------------------------------------------------------------
# Data Processor
# ------------------------------------------------------------

class DataProcessor:
    """
    Responsible for:
    - Encoding categorical features
    - Handling missing values
    - Standard scaling
    - Target binarisation
    """

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_means = None
        self.feature_stds = None

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:

        categorical_cols = X.select_dtypes(include="object").columns

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le

        return X

    def _handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:

        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in [np.float64, np.int64]:
                    X[col].fillna(X[col].mean(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)

        return X

    def _scale(self, X: pd.DataFrame) -> np.ndarray:

        self.feature_means = X.mean()
        self.feature_stds = X.std().replace(0, 1)

        return self.scaler.fit_transform(X)

    def process(self, X: pd.DataFrame) -> np.ndarray:

        logger.info("Processing dataset")

        X = X.copy()
        X = self._encode(X)
        X = self._handle_missing(X)
        X = self._scale(X)

        logger.info("Dataset processing completed")

        return X


# ------------------------------------------------------------
# Data Visualizer
# ------------------------------------------------------------

class DataVisualizer:
    """
    Full EDA Visualisation Suite
    """

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target
        plt.style.use(Config.STYLE)

    # --------------------------------------------------------
    def visualize_all(self):

        logger.info("Generating EDA visualisations")

        self._plot_target_distribution()
        self._plot_correlation_heatmap()
        self._plot_numeric_distributions()
        self._plot_categorical_vs_target()
        self._plot_boxplots()
        self._plot_kde()
        self._plot_violin()
        self._plot_3d_scatter()

        logger.info("EDA visualisations saved")

    # --------------------------------------------------------
    def _plot_target_distribution(self):

        counts = self.target.value_counts().sort_index()

        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        sns.barplot(x=counts.index, y=counts.values)
        plt.title("Loan Status Distribution")
        plt.savefig("SVM\graphs\eda_target_distribution.png")
        plt.close()

    # --------------------------------------------------------
    def _plot_correlation_heatmap(self):

        df = self.data.select_dtypes(include=[np.number]).copy()
        df[Config.TARGET_COLUMN] = self.target

        plt.figure(figsize=(14, 10), dpi=Config.DPI)
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("SVM\graphs\eda_correlation_heatmap.png")
        plt.close()

    # --------------------------------------------------------
    def _plot_numeric_distributions(self):

        num_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Distribution: {col}")
            plt.savefig(f"SVM\graphs\eda_numeric_{col}.png")
            plt.close()

    # --------------------------------------------------------
    def _plot_categorical_vs_target(self):

        cat_cols = self.data.select_dtypes(include="object").columns

        for col in cat_cols:
            plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
            ct = pd.crosstab(self.data[col], self.target, normalize="index")
            ct.plot(kind="bar", stacked=True)
            plt.title(f"{col} vs Target")
            plt.savefig(f"SVM\graphs\eda_cat_{col}.png")
            plt.close()

    # --------------------------------------------------------
    def _plot_boxplots(self):

        num_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
            sns.boxplot(x=self.target, y=self.data[col])
            plt.title(f"Boxplot: {col}")
            plt.savefig(f"SVM\graphs\eda_box_{col}.png")
            plt.close()

    # --------------------------------------------------------
    def _plot_kde(self):

        num_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
            for cls in self.target.unique():
                sns.kdeplot(self.data[self.target == cls][col], label=f"Class {cls}")
            plt.legend()
            plt.title(f"KDE: {col}")
            plt.savefig(f"SVM\graphs\eda_kde_{col}.png")
            plt.close()

    # --------------------------------------------------------
    def _plot_violin(self):

        num_cols = self.data.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
            sns.violinplot(x=self.target, y=self.data[col])
            plt.title(f"Violin: {col}")
            plt.savefig(f"SVM\graphs\eda_violin_{col}.png")
            plt.close()

    # --------------------------------------------------------
    def _plot_3d_scatter(self):

        from mpl_toolkits.mplot3d import Axes3D

        num_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(num_cols) < 3:
            return

        cols = num_cols[:3]

        fig = plt.figure(figsize=(10, 8), dpi=Config.DPI)
        ax = fig.add_subplot(111, projection="3d")

        for cls in self.target.unique():
            mask = self.target == cls
            ax.scatter(
                self.data[mask][cols[0]],
                self.data[mask][cols[1]],
                self.data[mask][cols[2]],
                label=f"Class {cls}",
                alpha=0.6,
            )

        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])
        ax.legend()
        plt.title("3D Scatter Plot")
        plt.savefig("SVM\graphs\eda_3d_scatter.png")
        plt.close()

# ============================================================
# 4. MODEL LAYER
# ============================================================

# ------------------------------------------------------------
# Core SVM
# ------------------------------------------------------------

class CoreSVM:
    """
    Core SVM Model
    Responsible only for:
    - Training
    - Prediction
    - Probability prediction
    - Saving and loading model
    """

    def __init__(self):
        self.model = SVC(**Config.SVM_PARAMS)
        self.is_trained = False

    def train(self, X_train, y_train):

        logger.info("Training SVM model")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        if hasattr(self.model, "support_vectors_"):
            logger.info(f"Number of support vectors: {len(self.model.support_vectors_)}")

    def predict(self, X):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        return self.model.predict(X)

    def predict_proba(self, X):

        if not self.is_trained:
            raise RuntimeError("Model not trained")

        probs = self.model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 else probs.ravel()

    def save(self):

        joblib.dump(self.model, Config.MODEL_PATH)
        logger.info(f"Model saved to {Config.MODEL_PATH}")

    def load(self):

        self.model = joblib.load(Config.MODEL_PATH)
        self.is_trained = True
        logger.info("Model loaded successfully")


# ------------------------------------------------------------
# Cross Validator
# ------------------------------------------------------------

class CrossValidator:
    """
    Responsible only for cross validation logic and plotting.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def perform(self):

        logger.info("Performing Stratified K Fold Cross Validation")

        cv = StratifiedKFold(
            n_splits=Config.CV_FOLDS,
            shuffle=True,
            random_state=Config.RANDOM_STATE,
        )

        svm = SVC(**Config.SVM_PARAMS)

        metrics = {
            "accuracy": cross_val_score(svm, self.X, self.y, cv=cv, scoring="accuracy"),
            "precision": cross_val_score(svm, self.X, self.y, cv=cv, scoring="precision"),
            "recall": cross_val_score(svm, self.X, self.y, cv=cv, scoring="recall"),
            "f1": cross_val_score(svm, self.X, self.y, cv=cv, scoring="f1"),
            "roc_auc": cross_val_score(svm, self.X, self.y, cv=cv, scoring="roc_auc"),
        }

        for name, scores in metrics.items():
            logger.info(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        self._plot_cv_results(metrics)

        return metrics

    def _plot_cv_results(self, metrics):

        plt.figure(figsize=(10, 5), dpi=Config.DPI)

        fold_labels = [f"Fold {i+1}" for i in range(Config.CV_FOLDS)]
        x = np.arange(Config.CV_FOLDS)
        width = 0.15

        for i, (metric, scores) in enumerate(metrics.items()):
            plt.bar(x + i * width, scores, width, label=metric)

        plt.xticks(x + width * 2, fold_labels)
        plt.ylim(0, 1.1)
        plt.ylabel("Score")
        plt.title("Cross Validation Results")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("SVM\graphs\model_cross_validation.png")
        plt.close()

        logger.info("Cross validation plot saved")


# ------------------------------------------------------------
# Hyperparameter Tuner
# ------------------------------------------------------------

class HyperparameterTuner:
    """
    Responsible only for GridSearchCV tuning and heatmap visualisation.
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def tune(self):

        logger.info("Starting GridSearchCV hyperparameter tuning")

        base_model = SVC(
            probability=True,
            random_state=Config.RANDOM_STATE
        )

        grid = GridSearchCV(
            base_model,
            Config.GRID_PARAMS,
            cv=Config.TUNE_FOLDS,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(self.X_train, self.y_train)

        logger.info(f"Best Parameters: {grid.best_params_}")
        logger.info(f"Best ROC AUC: {grid.best_score_:.4f}")

        self._plot_heatmap(grid)

        best_model = CoreSVM()
        best_model.model = grid.best_estimator_
        best_model.is_trained = True

        return best_model

    def _plot_heatmap(self, grid):

        results = pd.DataFrame(grid.cv_results_)

        rbf_mask = results["param_kernel"] == "rbf"

        if rbf_mask.sum() == 0:
            return

        pivot = results[rbf_mask].pivot_table(
            index="param_C",
            columns="param_gamma",
            values="mean_test_score",
        )

        plt.figure(figsize=(8, 5), dpi=Config.DPI)
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title("GridSearchCV ROC AUC Heatmap (RBF)")
        plt.tight_layout()
        plt.savefig("SVM\graphs\model_gridsearch_heatmap.png")
        plt.close()

        logger.info("GridSearch heatmap saved")


# ============================================================
# 5. EVALUATION LAYER
# ============================================================

# ------------------------------------------------------------
# Metrics Calculator
# ------------------------------------------------------------

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class MetricsCalculator:
    """
    Responsible only for computing evaluation metrics.
    No plotting logic here.
    """

    @staticmethod
    def compute(y_true, y_pred, y_prob) -> Metrics:

        return Metrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_prob),
        )


# ------------------------------------------------------------
# Evaluation Visualizer
# ------------------------------------------------------------

class EvaluationVisualizer:
    """
    Responsible only for evaluation plots:
    - Confusion Matrix
    - ROC Curve
    - Probability Distribution
    - Metrics Bar Chart
    - Calibration Plot
    - Confidence Analysis
    """

    def __init__(self):
        plt.style.use(Config.STYLE)

    # --------------------------------------------------------
    def visualize_all(self, y_true, y_pred, y_prob, dataset_name="Test"):

        logger.info(f"Generating evaluation plots for {dataset_name}")

        self._plot_confusion_matrix(y_true, y_pred, dataset_name)
        self._plot_roc_curve(y_true, y_prob, dataset_name)
        self._plot_probability_distribution(y_true, y_prob, dataset_name)
        self._plot_metrics_bar(y_true, y_pred, y_prob, dataset_name)
        self._plot_calibration(y_true, y_prob, dataset_name)
        self._plot_confidence_analysis(y_true, y_pred, y_prob, dataset_name)

        logger.info("Evaluation plots saved")

    # --------------------------------------------------------
    def _plot_confusion_matrix(self, y_true, y_pred, name):

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"SVM\graphs\eval_confusion_{name}.png")
        plt.close()

    # --------------------------------------------------------
    def _plot_roc_curve(self, y_true, y_prob, name):

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"SVM\graphs\eval_roc_{name}.png")
        plt.close()

    # --------------------------------------------------------
    def _plot_probability_distribution(self, y_true, y_prob, name):

        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)

        plt.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Rejected")
        plt.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Approved")

        plt.legend()
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(f"Probability Distribution - {name}")
        plt.tight_layout()
        plt.savefig(f"SVM\graphs\eval_probability_{name}.png")
        plt.close()

    # --------------------------------------------------------
    def _plot_metrics_bar(self, y_true, y_pred, y_prob, name):

        metrics = MetricsCalculator.compute(y_true, y_pred, y_prob)

        values = [
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.roc_auc,
        ]

        labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]

        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        bars = plt.bar(labels, values)
        plt.ylim(0, 1.1)
        plt.title(f"Performance Metrics - {name}")
        plt.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(f"SVM\graphs\eval_metrics_{name}.png")
        plt.close()

    # --------------------------------------------------------
    def _plot_calibration(self, y_true, y_prob, name):

        bins = np.linspace(0, 1, 11)
        binids = np.digitize(y_prob, bins) - 1

        true_probs = []
        pred_probs = []

        for i in range(len(bins) - 1):
            mask = binids == i
            if np.sum(mask) > 0:
                true_probs.append(np.mean(y_true[mask]))
                pred_probs.append(np.mean(y_prob[mask]))

        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
        plt.plot(pred_probs, true_probs, marker="o", label="SVM")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Plot - {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"SVM\graphs\eval_calibration_{name}.png")
        plt.close()

    # --------------------------------------------------------
    def _plot_confidence_analysis(self, y_true, y_pred, y_prob, name):

        correct = y_prob[y_true == y_pred]
        incorrect = y_prob[y_true != y_pred]

        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        plt.hist(correct, bins=20, alpha=0.6, label="Correct")
        plt.hist(incorrect, bins=20, alpha=0.6, label="Incorrect")
        plt.legend()
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(f"Prediction Confidence - {name}")
        plt.tight_layout()
        plt.savefig(f"SVM\graphs\eval_confidence_{name}.png")
        plt.close()

# ============================================================
# 6. ORCHESTRATION LAYER
# ============================================================

class TrainingPipeline:
    """
    Orchestrates entire workflow:
    - Load
    - Validate
    - Visualize
    - Process
    - Split
    - Train
    - Evaluate
    - Cross Validate
    - Hyperparameter Tune
    - New Applicant Simulation
    """

    def __init__(self):

        self.loader = DataLoader(Config.DATASET_PATH)
        self.processor = DataProcessor()
        self.model = CoreSVM()
        self.eval_visualizer = EvaluationVisualizer()

    # --------------------------------------------------------
    def run(self):

        logger.info("========== SVM PIPELINE STARTED ==========")

        # 1. Load
        X_raw, y_raw = self.loader.load()

        # 2. Validate
        validator = DataValidator(X_raw, y_raw)
        validator.validate()

        # 3. EDA
        visualizer = DataVisualizer(X_raw, y_raw)
        visualizer.visualize_all()

        # 4. Process
        X_processed = self.processor.process(X_raw)

        # 5. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y_raw,
            test_size=Config.TEST_SIZE,
            stratify=y_raw,
            random_state=Config.RANDOM_STATE,
        )

        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test shape: {X_test.shape}")

        # 6. Train
        self.model.train(X_train, y_train)
        self.model.save()

        # 7. Evaluate Train
        y_train_pred = self.model.predict(X_train)
        y_train_prob = self.model.predict_proba(X_train)

        train_metrics = MetricsCalculator.compute(
            y_train, y_train_pred, y_train_prob
        )

        logger.info(f"Training Accuracy: {train_metrics.accuracy:.4f}")

        self.eval_visualizer.visualize_all(
            y_train, y_train_pred, y_train_prob, "Train"
        )

        # 8. Evaluate Test
        y_test_pred = self.model.predict(X_test)
        y_test_prob = self.model.predict_proba(X_test)

        test_metrics = MetricsCalculator.compute(
            y_test, y_test_pred, y_test_prob
        )

        logger.info(f"Test Accuracy: {test_metrics.accuracy:.4f}")

        self.eval_visualizer.visualize_all(
            y_test, y_test_pred, y_test_prob, "Test"
        )

        # 9. Cross Validation
        cv = CrossValidator(X_processed, y_raw)
        cv.perform()

        # 10. Hyperparameter Tuning
        tuner = HyperparameterTuner(X_train, y_train)
        tuned_model = tuner.tune()

        logger.info("Evaluating tuned model")

        tuned_pred = tuned_model.predict(X_test)
        tuned_prob = tuned_model.predict_proba(X_test)

        tuned_metrics = MetricsCalculator.compute(
            y_test, tuned_pred, tuned_prob
        )

        logger.info(f"Tuned Test ROC AUC: {tuned_metrics.roc_auc:.4f}")

        # 11. New Applicant Simulation
        self._simulate_new_applicant(X_train.shape[1])

        logger.info("========== SVM PIPELINE COMPLETED ==========")

    # --------------------------------------------------------
    def _simulate_new_applicant(self, n_features):

        logger.info("Simulating new loan applicant prediction")

        means = self.processor.feature_means.values
        stds = self.processor.feature_stds.values.copy()
        stds[stds == 0] = 1.0

        raw = means + np.random.randn(n_features) * stds
        scaled = ((raw - means) / stds).reshape(1, -1)

        prob = self.model.predict_proba(scaled)[0]
        pred = self.model.predict(scaled)[0]

        decision = "Approved" if pred == 1 else "Rejected"

        logger.info(f"New Applicant Probability: {prob:.4f}")
        logger.info(f"Decision: {decision}")


# ============================================================
# 7. ENTRY POINT
# ============================================================

def main():
    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()