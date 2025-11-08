import os
import yaml
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_feature(stat: str, feature_dir: str):
    matrix_path = os.path.join(feature_dir, f'feature_matrix_{stat}.npy')
    meta_path = os.path.join(feature_dir, f'feature_matrix_{stat}_meta.npz')
    if not (os.path.exists(matrix_path) and os.path.exists(meta_path)):
        return None, None, None
    X = np.load(matrix_path)
    meta = np.load(meta_path, allow_pickle=True)
    dhs_sites = meta['dhs_sites']
    disease_labels = meta['disease_labels'] if 'disease_labels' in meta else None
    if disease_labels is None and 'binary_labels' in meta:
        disease_labels = meta['binary_labels']
    return X, dhs_sites, disease_labels


def plot_stat(matrix, dhs_sites, disease_labels, stat_name, out_dir):
    unique_dhs = np.unique(dhs_sites)
    for dhs in unique_dhs:
        dhs_mask = dhs_sites == dhs
        filtered_matrix = matrix[dhs_mask]
        filtered_diseases = disease_labels[dhs_mask]

        if stat_name == 'pfe':
            plt.figure(figsize=(10, 6))
            distributions = []
            disease_order = []
            for disease in sorted(np.unique(filtered_diseases)):
                d_mask = filtered_diseases == disease
                if d_mask.any():
                    distributions.append(filtered_matrix[d_mask].flatten())
                    disease_order.append(disease)
            if not distributions:
                plt.close()
                continue
            plt.violinplot(distributions, showmeans=False, showmedians=True)
            plt.title(f'PFE Distribution by Disease (DHS={dhs})')
            plt.xlabel('Disease')
            plt.ylabel('PFE value')
            plt.xticks(range(1, len(disease_order) + 1), disease_order, rotation=30, ha='right')
            plt.tight_layout()
            out_path = os.path.join(out_dir, f'{stat_name}_{dhs}_violin_disease.png')
            plt.savefig(out_path, dpi=200)
            plt.close()
            continue

        if filtered_matrix.shape[0] < 2 or filtered_matrix.shape[1] < 2:
            continue

        pca = PCA(n_components=min(10, filtered_matrix.shape[1], filtered_matrix.shape[0]))
        X_pca = pca.fit_transform(filtered_matrix)
        expl_var = pca.explained_variance_ratio_

        plt.figure(figsize=(8, 6))
        for disease in np.unique(filtered_diseases):
            d_mask = filtered_diseases == disease
            plt.scatter(X_pca[d_mask, 0], X_pca[d_mask, 1], label=disease, alpha=0.55)
        plt.xlabel(f'PC1 ({expl_var[0]*100:.1f}% var)')
        plt.ylabel(f'PC2 ({expl_var[1]*100:.1f}% var)')
        plt.title(f'PCA of {stat_name.upper()} (DHS={dhs})')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        out_path_scatter = os.path.join(out_dir, f'{stat_name}_{dhs}_pca_by_disease.png')
        plt.savefig(out_path_scatter, dpi=200)
        plt.close()

        pc1 = pca.components_[0]
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(pc1)), pc1)
        plt.axvline(x=len(pc1)//2, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Feature Index')
        plt.ylabel('Weight in PC1')
        plt.title(f'PC1 Feature Weights ({stat_name.upper()}, DHS={dhs})')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        out_path_pc1 = os.path.join(out_dir, f'{stat_name}_{dhs}_pc1_weights.png')
        plt.savefig(out_path_pc1, dpi=200)
        plt.close()


if 'snakemake' in globals():
    config_path = snakemake.input.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    feature_dir = config['result_dir']
    out_dir = config['result_pca_dir']
    stats = snakemake.params.stats

    os.makedirs(out_dir, exist_ok=True)

    for stat in stats:
        X, dhs_sites, disease_labels = load_feature(stat, feature_dir)
        if X is None:
            print(f"[skip] {stat}: feature matrix/meta not found in {feature_dir}")
            continue
        print(f"Processing: {stat}")
        plot_stat(X, dhs_sites, disease_labels, stat, out_dir)

    if hasattr(snakemake, 'output') and 'done' in snakemake.output:
        with open(snakemake.output.done, 'w') as fh:
            fh.write('ok\n')
