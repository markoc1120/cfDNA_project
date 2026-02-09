import os
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_feature_df(stat: str, feature_dir: str) -> pd.DataFrame:
    parquet_path = os.path.join(feature_dir, f'feature_matrix_{stat}.parquet')
    if not os.path.exists(parquet_path):
        print(f'Missing parquet: {parquet_path}')
        return None

    return pd.read_parquet(parquet_path)


def plot_stat(df: pd.DataFrame, stat_name: str, out_dir: str):
    for dhs in df['pc1'].columns:
        if stat_name == 'pfe':
            data = df['pc1'][dhs].reset_index()
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=data, x='disease', y=dhs, inner='quart')
            plt.title(f'PFE Distribution by Disease (stat={stat}, DHS={dhs})')
            plt.xlabel('Disease')
            plt.ylabel('PFE value')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            out_path = os.path.join(out_dir, f'{stat_name}_{dhs}_violin_disease.png')
            plt.savefig(out_path, dpi=200)
            plt.close()
            continue
        
        modified_df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            modified_df.columns = [f"{pc}_{dhs}" for pc, dhs in modified_df.columns]
        
        # pc1 vs pc2
        plt.figure(figsize=(10, 6))
        sns.scatterplot(modified_df, x=f'pc1_{dhs}', y=f'pc2_{dhs}', hue='disease', alpha=0.55)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'PC1 vs PC2 (stat={stat}, DHS={dhs})')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        out_path_scatter = os.path.join(out_dir, f'{stat_name}_{dhs}_pca_by_disease.png')
        plt.savefig(out_path_scatter, dpi=200)
        plt.close()
        

def plot_loadings(stat_name: str, result_dir: str, out_dir: str):
    if stat_name == 'pfe':
        return
    loadings_path = os.path.join(result_dir, f'{stat_name}_pca_loadings.csv')
    loadings_df = pd.read_csv(loadings_path)
              
    # pc1 weights
    plt.figure(figsize=(10, 6))
    plt.plot(loadings_df['PC1'])
    num_points = loadings_df.shape[0]
    plt.axvline(x=num_points//2, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Feature Index')
    plt.ylabel('Weight in PC1')
    plt.title(f'PC1 Feature Weights (stat={stat_name})')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path_pc1 = os.path.join(out_dir, f'{stat_name}_pc1_loadings.png')
    plt.savefig(out_path_pc1, dpi=200)
    plt.close()


if 'snakemake' in globals():
    config_path = snakemake.input.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    result_dir = config['result_dir']
    feature_dir = config['result_dir']
    out_dir = config['result_pca_dir']
    stats = snakemake.params.stats

    os.makedirs(out_dir, exist_ok=True)

    for stat in stats:
        df = load_feature_df(stat, feature_dir)
        if df is None or df.empty:
            print(f"{stat}: No parquet data found.")
            continue
        print(f"Processing: {stat}")
        plot_stat(df, stat, out_dir)
        plot_loadings(stat, result_dir, out_dir)

    if hasattr(snakemake, 'output') and 'done' in snakemake.output:
        with open(snakemake.output.done, 'w') as fh:
            fh.write('ok\n')
