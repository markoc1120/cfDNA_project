import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def load_stat(stat: str, feature_dir: str):
    matrix_path = os.path.join(feature_dir, f"feature_matrix_{stat}.npy")
    meta_path = os.path.join(feature_dir, f"feature_matrix_{stat}_meta.npz")
    if not (os.path.exists(matrix_path) and os.path.exists(meta_path)):
        return None
    X = np.load(matrix_path)
    meta = np.load(meta_path)

    labels = meta['binary_labels'] if 'binary_labels' in meta else None
    return X, labels


def evaluate_stat_pc1(stat: str, feature_dir: str, cv_splits: int):
    loaded = load_stat(stat, feature_dir)
    if loaded is None:
        return None
    X, labels = loaded
    if labels is None:
        return None

    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    n_pca_components = 1
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_pca_components)),
        ('classifier', SVC(probability=True, random_state=42)),
    ])

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    return {
        'stat': stat,
        'auc_mean': float(np.mean(scores)),
        'auc_std': float(np.std(scores)),
    }


if 'snakemake' in globals():
    config_path = snakemake.input.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    feature_dir = config['result_dir']
    out_dir = config['result_pca_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    stats = snakemake.params.stats
    cv_splits = 10

    results = []
    for stat in stats:
        r = evaluate_stat_pc1(stat, feature_dir, cv_splits)
        if r is not None:
            results.append(r)

    if results:
        stats_order = [r['stat'] for r in results]
        auc_means = [r['auc_mean'] for r in results]
        auc_stds = [r['auc_std'] for r in results]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(stats_order, auc_means, yerr=auc_stds, capsize=5)
        plt.ylabel('Mean ROC AUC (PC1 across DHS)')
        plt.xlabel('Test statistic')
        for bar, val in zip(bars, auc_means):
            plt.text(
                bar.get_x() + bar.get_width()/2 - 0.05,
                bar.get_height(), 
                f"{val:.2f}", 
                ha='right', 
                va='bottom', 
                fontsize=9
            )
        plt.title('Binary classification result (PC1 features + SVM)')
        plt.tight_layout()
        plt.savefig(snakemake.output.auc_plot, dpi=200)
        plt.close()
