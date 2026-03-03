import warnings
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import seaborn as sns
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# --- Global Variables ---------------------------------------------------------
RANDOM_STATE    = 42
DATASET_PATH    = "income.csv"
N_CLUSTERS      = 3           # Number of clusters for KMeans
MAX_ITER        = 300         # Maximum iterations for KMeans
N_INIT          = 10          # Number of initializations
INIT_METHOD     = 'k-means++' # 'k-means++' | 'random'
ELBOW_MAX_K     = 10          # Max K to test in elbow method

FEATURE_COLUMNS = ['Age', 'Income($)']
ID_COLUMN       = 'Name'

FIGURE_SIZE     = (12, 8)
DPI             = 100
STYLE           = 'seaborn-v0_8-darkgrid'

CLUSTER_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# --- Data Class ---------------------------------------------------------------
@dataclass
class ClusterMetrics:
    """Stores cluster evaluation metrics."""
    inertia:            float
    silhouette:         float
    davies_bouldin:     float
    calinski_harabasz:  float

    def __str__(self) -> str:
        return (
            f"Cluster Evaluation Metrics:\n"
            f"{'=' * 50}\n"
            f"Inertia (WCSS)        : {self.inertia:.4f}\n"
            f"Silhouette Score      : {self.silhouette:.4f}  (higher is better, range -1 to 1)\n"
            f"Davies-Bouldin Index  : {self.davies_bouldin:.4f}  (lower is better)\n"
            f"Calinski-Harabasz     : {self.calinski_harabasz:.4f}  (higher is better)\n"
            f"{'=' * 50}"
        )


# --- DatasetLoader ------------------------------------------------------------
class DatasetLoader:
    """Loads income data from CSV or generates a synthetic fallback (600 rows)."""

    def __init__(self, dataset_path: str = None):
        self.dataset_path  = dataset_path
        self.data: Optional[pd.DataFrame] = None
        self.feature_names: Optional[list] = None

    def _generate_synthetic_dataset(self, n_samples: int = 600) -> pd.DataFrame:
        """
        Generate synthetic income dataset with 3 realistic clusters:
          Cluster 0 - Young low-income  (Age 22-32, Income 40k-75k)
          Cluster 1 - Mid-age mid-income (Age 33-45, Income 120k-165k)
          Cluster 2 - Senior high-income (Age 38-55, Income 55k-90k)
        """
        np.random.seed(RANDOM_STATE)

        first_names = [
            'Rob', 'Michael', 'Mohan', 'Ismail', 'Kory', 'Gautam', 'David',
            'Andrea', 'Brad', 'Angelina', 'Donald', 'Tom', 'Arnold', 'Jared',
            'Stark', 'Ranbir', 'Dipika', 'Priyanka', 'Nick', 'Alia', 'Sid',
            'Abdul', 'Emma', 'Liam', 'Olivia', 'Noah', 'Sophia', 'James',
            'Isabella', 'Oliver', 'Mia', 'Elijah', 'Charlotte', 'William',
            'Amelia', 'Benjamin', 'Harper', 'Lucas', 'Evelyn', 'Mason',
            'Abigail', 'Ethan', 'Emily', 'Alexander', 'Elizabeth', 'Henry',
            'Mila', 'Jackson', 'Ella', 'Sebastian', 'Avery', 'Aiden', 'Sofia',
            'Matthew', 'Camila', 'Samuel', 'Aria', 'David', 'Scarlett', 'Joseph',
            'Victoria', 'Carter', 'Madison', 'Owen', 'Luna', 'Wyatt', 'Grace',
            'John', 'Chloe', 'Jack', 'Penelope', 'Luke', 'Layla', 'Jayden',
            'Riley', 'Dylan', 'Zoey', 'Grayson', 'Nora', 'Levi', 'Lily',
            'Isaac', 'Eleanor', 'Gabriel', 'Hannah', 'Julian', 'Lillian',
            'Mateo', 'Addison', 'Anthony', 'Aubrey', 'Jaxon', 'Ellie', 'Lincoln',
            'Stella', 'Joshua', 'Natalie', 'Christopher', 'Zoe', 'Andrew', 'Leah',
        ]

        sizes   = [n_samples // 3, n_samples // 3, n_samples - 2 * (n_samples // 3)]
        records = []

        # Cluster 0: Young, low income
        ages_0    = np.random.randint(22, 33, sizes[0])
        incomes_0 = np.random.randint(40000, 76000, sizes[0])
        for a, i in zip(ages_0, incomes_0):
            records.append({'Age': int(a), 'Income($)': int(i), 'TrueCluster': 0})

        # Cluster 1: Mid-age, mid income
        ages_1    = np.random.randint(33, 46, sizes[1])
        incomes_1 = np.random.randint(120000, 166000, sizes[1])
        for a, i in zip(ages_1, incomes_1):
            records.append({'Age': int(a), 'Income($)': int(i), 'TrueCluster': 1})

        # Cluster 2: Senior, high income (overlap zone)
        ages_2    = np.random.randint(38, 56, sizes[2])
        incomes_2 = np.random.randint(55000, 91000, sizes[2])
        for a, i in zip(ages_2, incomes_2):
            records.append({'Age': int(a), 'Income($)': int(i), 'TrueCluster': 2})

        df = pd.DataFrame(records)

        # Assign names (cycle through list if needed)
        name_pool = first_names * (n_samples // len(first_names) + 2)
        df[ID_COLUMN] = name_pool[:n_samples]
        df = df[[ID_COLUMN, 'Age', 'Income($)', 'TrueCluster']].reset_index(drop=True)
        return df

    def load_data(self) -> pd.DataFrame:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                required = set(FEATURE_COLUMNS)
                if required.issubset(set(df.columns)):
                    self.data = df
                    self.feature_names = FEATURE_COLUMNS
                    print(f"[OK] Loaded dataset from: {self.dataset_path}")
                    print(f"[OK] Samples  : {len(self.data)}")
                    print(f"[OK] Features : {self.feature_names}")
                    return self.data
                else:
                    print(f"[WARN] Expected columns {required} not found.")
            except Exception as ex:
                print(f"[WARN] Failed to load {self.dataset_path}: {ex}")
            print("[WARN] Falling back to synthetic dataset generation")

        self.data = self._generate_synthetic_dataset(n_samples=600)
        self.feature_names = FEATURE_COLUMNS
        print(f"[OK] Synthetic dataset generated")
        print(f"[OK] Samples  : {len(self.data)}")
        print(f"[OK] Features : {self.feature_names}")
        print(f"[OK] Columns  : {list(self.data.columns)}")
        return self.data


# --- DatasetValidator ---------------------------------------------------------
class DatasetValidator:
    """Validates and summarises the loaded dataset."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        if self.data.empty:
            print("[ERROR] Dataset is empty!")
            return False
        print("[OK] Dataset is not empty")

        print(f"\n--- Shape ---")
        print(f"Rows: {self.data.shape[0]}  |  Columns: {self.data.shape[1]}")

        print(f"\n--- Missing Values ---")
        miss = self.data.isnull().sum()
        print(miss[miss > 0] if miss.sum() > 0 else "No missing values detected")

        print(f"\n--- Data Types ---")
        print(self.data.dtypes)

        print(f"\n--- First 5 Rows ---")
        print(self.data.head())

        print(f"\n--- Statistical Summary ---")
        print(self.data[FEATURE_COLUMNS].describe())

        return True


# --- DatasetProcessor ---------------------------------------------------------
class DatasetProcessor:
    """
    Scales features for KMeans.

    NOTE: Unlike Decision Trees, KMeans is a distance-based algorithm.
    Feature scaling (StandardScaler) is REQUIRED so that features with
    larger numeric ranges do not dominate the distance calculation.
    """

    def __init__(self, data: pd.DataFrame):
        self.data            = data.copy()
        self.scaler          = StandardScaler()
        self.X_raw:  Optional[np.ndarray] = None
        self.X_scaled: Optional[np.ndarray] = None

    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        print("\n--- Handling Missing Values ---")
        before = self.data[FEATURE_COLUMNS].isnull().sum().sum()
        for col in FEATURE_COLUMNS:
            self.data[col].fillna(self.data[col].median(), inplace=True)
        print(f"Missing before: {before}  ->  after: {self.data[FEATURE_COLUMNS].isnull().sum().sum()}")

        self.X_raw = self.data[FEATURE_COLUMNS].values

        print("\n--- StandardScaler (Required for KMeans) ---")
        print("  NOTE: KMeans uses Euclidean distance, so scaling is mandatory.")
        print("        Without scaling, Income($) would dominate Age entirely.")
        self.X_scaled = self.scaler.fit_transform(self.X_raw)

        print(f"\n  Feature means  (pre-scale)  : {self.scaler.mean_}")
        print(f"  Feature stdevs (pre-scale)  : {np.sqrt(self.scaler.var_)}")
        print(f"\n[OK] Features scaled  ->  shape {self.X_scaled.shape}")
        return self.X_raw, self.X_scaled


# --- IncomeVisualizer ---------------------------------------------------------
class IncomeVisualizer:
    """Produces exploratory visualisations for the income dataset."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        plt.style.use(STYLE)

    def visualize(self):
        print(f"\n{'=' * 70}")
        print("INCOME DATASET VISUALIZATION")
        print(f"{'=' * 70}")

        self.plot_missing_values()
        self.plot_feature_distributions()
        self.plot_age_vs_income()
        self.plot_income_boxplot()
        self.plot_age_boxplot()
        self.plot_correlation_heatmap()

        print("[OK] All exploratory visualisations saved")

    def plot_missing_values(self):
        if self.data.isnull().sum().sum() == 0:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=DPI)
            ax.text(0.5, 0.5, 'No missing values in dataset',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
            sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Heatmap', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("income_missing_values.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Missing values plot saved")

    def plot_feature_distributions(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        for ax, col in zip(axes, FEATURE_COLUMNS):
            ax.hist(self.data[col], bins=30, color='#1f77b4',
                    edgecolor='black', alpha=0.8)
            ax.axvline(self.data[col].mean(), color='red', linestyle='--',
                       label=f'Mean: {self.data[col].mean():.1f}')
            ax.axvline(self.data[col].median(), color='green', linestyle='-.',
                       label=f'Median: {self.data[col].median():.1f}')
            ax.set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("income_feature_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature distributions saved")

    def plot_age_vs_income(self):
        plt.figure(figsize=(10, 6), dpi=DPI)
        plt.scatter(self.data['Age'], self.data['Income($)'],
                    alpha=0.6, edgecolors='black', linewidths=0.3,
                    color='#1f77b4', s=60)
        plt.title("Age vs Income (Raw Data)", fontsize=12, fontweight='bold')
        plt.xlabel("Age")
        plt.ylabel("Income ($)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("income_age_vs_income_raw.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Age vs Income (raw) saved")

    def plot_income_boxplot(self):
        plt.figure(figsize=(8, 5), dpi=DPI)
        plt.boxplot(self.data['Income($)'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#1f77b4', alpha=0.7))
        plt.title("Income Distribution (Boxplot)", fontsize=12, fontweight='bold')
        plt.ylabel("Income ($)")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("income_boxplot.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Income boxplot saved")

    def plot_age_boxplot(self):
        plt.figure(figsize=(8, 5), dpi=DPI)
        plt.boxplot(self.data['Age'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#ff7f0e', alpha=0.7))
        plt.title("Age Distribution (Boxplot)", fontsize=12, fontweight='bold')
        plt.ylabel("Age")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("income_age_boxplot.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Age boxplot saved")

    def plot_correlation_heatmap(self):
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return
        plt.figure(figsize=(7, 5), dpi=DPI)
        sns.heatmap(self.data[num_cols].corr(), annot=True, cmap='coolwarm',
                    fmt='.2f', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Heatmap", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("income_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Correlation heatmap saved")


# --- KMeansModel --------------------------------------------------------------
class KMeansModel:
    """
    KMeans clustering wrapper.

    Key difference from Decision Tree
    ----------------------------------
    - Decision Tree: supervised, builds if/else rules from labelled data.
    - KMeans: unsupervised, partitions data by minimising within-cluster
      sum of squares (WCSS / inertia).  No labels are required.
    - KMeans REQUIRES feature scaling; Decision Trees do not.
    """

    def __init__(
        self,
        n_clusters: int  = N_CLUSTERS,
        init: str        = INIT_METHOD,
        max_iter: int    = MAX_ITER,
        n_init: int      = N_INIT,
    ):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters  = n_clusters,
            init        = init,
            max_iter    = max_iter,
            n_init      = n_init,
            random_state = RANDOM_STATE,
        )
        self.labels_:     Optional[np.ndarray] = None
        self.centroids_:  Optional[np.ndarray] = None

    def fit_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        print(f"\n{'=' * 60}")
        print("TRAINING KMEANS MODEL")
        print(f"{'=' * 60}")
        print(f"n_clusters  : {self.n_clusters}")
        print(f"init method : {self.model.init}")
        print(f"max_iter    : {self.model.max_iter}")
        print(f"n_init      : {self.model.n_init}")

        self.labels_    = self.model.fit_predict(X_scaled)
        self.centroids_ = self.model.cluster_centers_

        print(f"\n[OK] Clustering completed")
        print(f"     Iterations to converge : {self.model.n_iter_}")
        print(f"     Inertia (WCSS)          : {self.model.inertia_:.4f}")

        unique, counts = np.unique(self.labels_, return_counts=True)
        print(f"\n     Cluster sizes:")
        for cl, cnt in zip(unique, counts):
            print(f"       Cluster {cl} : {cnt} samples")

        return self.labels_

    def get_metrics(self, X_scaled: np.ndarray) -> ClusterMetrics:
        sil = silhouette_score(X_scaled, self.labels_)
        dbi = davies_bouldin_score(X_scaled, self.labels_)
        chi = calinski_harabasz_score(X_scaled, self.labels_)
        return ClusterMetrics(
            inertia           = self.model.inertia_,
            silhouette        = sil,
            davies_bouldin    = dbi,
            calinski_harabasz = chi,
        )


# --- ModelEvaluator -----------------------------------------------------------
class ModelEvaluator:
    """Generates evaluation and cluster visualisation plots."""

    def __init__(self, model: KMeansModel, scaler: StandardScaler):
        self.model  = model
        self.scaler = scaler

    def evaluate(self, X_raw: np.ndarray, X_scaled: np.ndarray, df: pd.DataFrame):
        print(f"\n{'=' * 70}")
        print("MODEL EVALUATION")
        print(f"{'=' * 70}")

        metrics = self.model.get_metrics(X_scaled)
        print(metrics)

        self._plot_clusters(X_raw, df)
        self._plot_centroids(X_raw, X_scaled)
        self._plot_cluster_distributions(df)
        self._plot_pca_projection(X_scaled)

    def _plot_clusters(self, X_raw: np.ndarray, df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=DPI)

        # Left: raw scatter
        for cl in range(self.model.n_clusters):
            mask = self.model.labels_ == cl
            axes[0].scatter(X_raw[mask, 0], X_raw[mask, 1],
                            label=f'Cluster {cl}',
                            color=CLUSTER_COLORS[cl],
                            alpha=0.7, edgecolors='black', linewidths=0.3, s=60)

        # Centroids in original scale
        c_orig = self.scaler.inverse_transform(self.model.centroids_)
        axes[0].scatter(c_orig[:, 0], c_orig[:, 1],
                        marker='X', s=250, c='black', zorder=5, label='Centroids')
        axes[0].set_title('KMeans Clusters (Age vs Income)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Income ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: metrics bar
        metrics = self.model.get_metrics(
            self.scaler.transform(X_raw)
        )
        metric_names   = ['Silhouette', 'Davies-Bouldin\n(inverted)', 'Calinski-\nHarabasz (scaled)']
        # Normalise CH for display
        ch_scaled = min(metrics.calinski_harabasz / 1000.0, 1.0)
        db_inv    = 1.0 / (1.0 + metrics.davies_bouldin)
        values    = [metrics.silhouette, db_inv, ch_scaled]
        colors_b  = ['#1f77b4', '#ff7f0e', '#2ca02c']

        bars = axes[1].bar(metric_names, values, color=colors_b,
                           edgecolor='black', alpha=0.8)
        axes[1].set_ylim([0, 1.1])
        axes[1].set_ylabel('Score')
        axes[1].set_title('Cluster Quality Metrics', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig("income_cluster_result.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Cluster result plot saved")

    def _plot_centroids(self, X_raw: np.ndarray, X_scaled: np.ndarray):
        c_orig = self.scaler.inverse_transform(self.model.centroids_)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        # Centroid positions
        for cl in range(self.model.n_clusters):
            mask = self.model.labels_ == cl
            axes[0].scatter(X_raw[mask, 0], X_raw[mask, 1],
                            color=CLUSTER_COLORS[cl], alpha=0.4,
                            edgecolors='black', linewidths=0.2, s=40)
        for idx, (cx, cy) in enumerate(c_orig):
            axes[0].scatter(cx, cy, marker='X', s=300,
                            color=CLUSTER_COLORS[idx], edgecolors='black',
                            linewidths=1.5, zorder=6,
                            label=f'Centroid {idx}: Age={cx:.1f}, Inc={cy:,.0f}')
        axes[0].set_title('Cluster Centroids', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Income ($)')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Centroid bar chart
        bar_width = 0.35
        x_idx = np.arange(self.model.n_clusters)
        b1 = axes[1].bar(x_idx - bar_width / 2, c_orig[:, 0], bar_width,
                         label='Age', color='#1f77b4', edgecolor='black', alpha=0.8)
        ax2 = axes[1].twinx()
        b2 = ax2.bar(x_idx + bar_width / 2, c_orig[:, 1], bar_width,
                     label='Income ($)', color='#ff7f0e', edgecolor='black', alpha=0.8)
        axes[1].set_xticks(x_idx)
        axes[1].set_xticklabels([f'Cluster {i}' for i in range(self.model.n_clusters)])
        axes[1].set_ylabel('Age (centroid)', color='#1f77b4')
        ax2.set_ylabel('Income $ (centroid)', color='#ff7f0e')
        axes[1].set_title('Centroid Values per Cluster', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig("income_centroids.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Centroid plot saved")

    def _plot_cluster_distributions(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

        # Age distribution per cluster
        for cl in range(self.model.n_clusters):
            mask = self.model.labels_ == cl
            axes[0][0].hist(df.loc[mask, 'Age'].values, bins=15,
                            alpha=0.6, label=f'Cluster {cl}',
                            color=CLUSTER_COLORS[cl], edgecolor='black')
        axes[0][0].set_title('Age Distribution by Cluster', fontsize=11, fontweight='bold')
        axes[0][0].set_xlabel('Age')
        axes[0][0].set_ylabel('Frequency')
        axes[0][0].legend()
        axes[0][0].grid(True, alpha=0.3)

        # Income distribution per cluster
        for cl in range(self.model.n_clusters):
            mask = self.model.labels_ == cl
            axes[0][1].hist(df.loc[mask, 'Income($)'].values, bins=15,
                            alpha=0.6, label=f'Cluster {cl}',
                            color=CLUSTER_COLORS[cl], edgecolor='black')
        axes[0][1].set_title('Income Distribution by Cluster', fontsize=11, fontweight='bold')
        axes[0][1].set_xlabel('Income ($)')
        axes[0][1].set_ylabel('Frequency')
        axes[0][1].legend()
        axes[0][1].grid(True, alpha=0.3)

        # Age boxplot per cluster
        cluster_data_age = [df.loc[self.model.labels_ == cl, 'Age'].values
                            for cl in range(self.model.n_clusters)]
        bp1 = axes[1][0].boxplot(cluster_data_age, patch_artist=True,
                                  labels=[f'Cluster {i}' for i in range(self.model.n_clusters)])
        for patch, color in zip(bp1['boxes'], CLUSTER_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1][0].set_title('Age Spread per Cluster', fontsize=11, fontweight='bold')
        axes[1][0].set_ylabel('Age')
        axes[1][0].grid(True, alpha=0.3, axis='y')

        # Income boxplot per cluster
        cluster_data_inc = [df.loc[self.model.labels_ == cl, 'Income($)'].values
                            for cl in range(self.model.n_clusters)]
        bp2 = axes[1][1].boxplot(cluster_data_inc, patch_artist=True,
                                  labels=[f'Cluster {i}' for i in range(self.model.n_clusters)])
        for patch, color in zip(bp2['boxes'], CLUSTER_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1][1].set_title('Income Spread per Cluster', fontsize=11, fontweight='bold')
        axes[1][1].set_ylabel('Income ($)')
        axes[1][1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("income_cluster_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Cluster distribution plots saved")

    def _plot_pca_projection(self, X_scaled: np.ndarray):
        """Project to 2D via PCA (useful when more than 2 features exist)."""
        pca   = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(9, 6), dpi=DPI)
        for cl in range(self.model.n_clusters):
            mask = self.model.labels_ == cl
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=f'Cluster {cl}', color=CLUSTER_COLORS[cl],
                        alpha=0.7, edgecolors='black', linewidths=0.3, s=60)

        # PCA centroids
        c_pca = pca.transform(self.model.centroids_)
        plt.scatter(c_pca[:, 0], c_pca[:, 1],
                    marker='X', s=250, c='black', zorder=5, label='Centroids')

        plt.title("PCA Projection of Clusters", fontsize=12, fontweight='bold')
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("income_pca_clusters.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] PCA projection plot saved")


# --- MLPipeline ---------------------------------------------------------------
class MLPipeline:
    """End-to-end KMeans clustering pipeline for Income data."""

    def __init__(self):
        self.loader    = DatasetLoader(DATASET_PATH)
        self.processor = None
        self.model     = None
        self.evaluator = None

    def run(self):
        print(f"\n{'=' * 70}")
        print("KMEANS PIPELINE --- INCOME CLUSTERING")
        print(f"{'=' * 70}")

        # 1. Load
        df = self.loader.load_data()

        # 2. Visualise raw data
        visualizer = IncomeVisualizer(df)
        visualizer.visualize()

        # 3. Validate
        validator = DatasetValidator(df)
        validator.verify_dataset()

        # 4. Process (scale features)
        self.processor = DatasetProcessor(df)
        X_raw, X_scaled = self.processor.process_dataset()

        # 5. Find optimal K via Elbow + Silhouette
        optimal_k = self._find_optimal_k(X_scaled)

        # 6. Train KMeans (use configured N_CLUSTERS, print optimal K for reference)
        print(f"\n[INFO] Optimal K suggested by elbow/silhouette: {optimal_k}")
        print(f"[INFO] Using N_CLUSTERS = {N_CLUSTERS} (set in global config)")

        self.model = KMeansModel(n_clusters=N_CLUSTERS)
        labels     = self.model.fit_predict(X_scaled)

        # 7. Attach cluster labels back to dataframe
        df['Cluster'] = labels
        self._print_cluster_summary(df, X_raw)

        # 8. Evaluate & plot
        self.evaluator = ModelEvaluator(self.model, self.processor.scaler)
        self.evaluator.evaluate(X_raw, X_scaled, df)

        # 9. Save labelled dataset
        output_path = "income_clustered.csv"
        cols_to_save = [c for c in df.columns if c != 'TrueCluster']
        df[cols_to_save].to_csv(output_path, index=False)
        print(f"\n[OK] Labelled dataset saved -> {output_path}")

        # 10. Predict a new person
        self._predict_new_person()

    # -- Find Optimal K --------------------------------------------------------
    def _find_optimal_k(self, X_scaled: np.ndarray) -> int:
        print(f"\n{'=' * 70}")
        print("FINDING OPTIMAL K (ELBOW + SILHOUETTE)")
        print(f"{'=' * 70}")

        k_range    = range(2, ELBOW_MAX_K + 1)
        inertias   = []
        sil_scores = []

        for k in k_range:
            km = KMeans(n_clusters=k, init=INIT_METHOD, n_init=N_INIT,
                        random_state=RANDOM_STATE)
            lbl = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, lbl))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        # Elbow
        axes[0].plot(k_range, inertias, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        axes[0].set_title('Elbow Method (WCSS vs K)', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters K')
        axes[0].set_ylabel('Inertia (WCSS)')
        axes[0].grid(True, alpha=0.3)

        # Silhouette
        best_k_idx = int(np.argmax(sil_scores))
        best_k     = list(k_range)[best_k_idx]
        axes[1].plot(k_range, sil_scores, 'o-', linewidth=2, markersize=8, color='#ff7f0e')
        axes[1].axvline(best_k, color='red', linestyle='--',
                        label=f'Best K = {best_k}')
        axes[1].set_title('Silhouette Score vs K', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters K')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("income_optimal_k.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Optimal-K plot saved")
        print(f"[OK] Best K by silhouette: {best_k}  (score: {sil_scores[best_k_idx]:.4f})")

        return best_k

    # -- Cluster Summary -------------------------------------------------------
    def _print_cluster_summary(self, df: pd.DataFrame, X_raw: np.ndarray):
        print(f"\n{'=' * 70}")
        print("CLUSTER SUMMARY")
        print(f"{'=' * 70}")

        for cl in sorted(df['Cluster'].unique()):
            subset = df[df['Cluster'] == cl]
            print(f"\n  Cluster {cl}  ({len(subset)} members):")
            print(f"    Age     : min={subset['Age'].min()}  "
                  f"max={subset['Age'].max()}  "
                  f"mean={subset['Age'].mean():.1f}")
            print(f"    Income  : min={subset['Income($)'].min():,}  "
                  f"max={subset['Income($)'].max():,}  "
                  f"mean={subset['Income($)'].mean():,.0f}")
            if ID_COLUMN in subset.columns:
                sample = subset[ID_COLUMN].head(5).tolist()
                print(f"    Sample names : {sample}")

    # -- Predict new person ----------------------------------------------------
    def _predict_new_person(self):
        print(f"\n{'=' * 70}")
        print("NEW PERSON PREDICTION")
        print(f"{'=' * 70}")

        new_person = {'Age': 38, 'Income($)': 72000}
        print("New person profile:")
        for k, v in new_person.items():
            print(f"  {k}: {v}")

        X_new       = np.array([[new_person['Age'], new_person['Income($)']]])
        X_new_scaled = self.processor.scaler.transform(X_new)
        cluster      = self.model.model.predict(X_new_scaled)[0]

        # Distance to each centroid
        distances = np.linalg.norm(
            X_new_scaled - self.model.centroids_, axis=1
        )

        print(f"\nPredicted Cluster : {cluster}")
        print(f"\nDistances to centroids (scaled space):")
        for cl, dist in enumerate(distances):
            bar = '#' * int((1.0 / (dist + 0.01)) * 10)
            print(f"  Cluster {cl} : {dist:.4f}  {bar}")


# --- Entry Point --------------------------------------------------------------
def main():
    """Run the KMeans Income Clustering pipeline."""
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed:")
        print(f"{type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    main()