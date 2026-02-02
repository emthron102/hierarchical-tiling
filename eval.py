import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

# --- Configuration ---
HUMAN_ROOT = './data'  # Ground Truth (Human)
NN_ROOT = './results'  # Predictions (System)
SUBDIRS = ['travel_guides', 'vlogs']

# Column mappings
# Adjust these keys if your HUMAN data has different column headers
HUMAN_COLS = {
    'coarse_binary': 'boundary-coarse',
    'coarse_id': 'segment-coarse',
    'fine_binary': 'boundary-fine',
    'fine_id': 'segment-fine'
}

NN_COLS = {
    'coarse_binary': 'boundary-coarse',
    'coarse_id': 'segment-coarse',
    'fine_binary': 'boundary-fine',
    'fine_id': 'segment-fine'
}

def get_tile_stats(df, col_binary, col_id):
    """
    Calculates average tile length in sentences.
    """
    # Group by Tile ID and count sentences (rows)
    # We filter out tile_ids that might be NaN just in case
    tile_counts = df.groupby(col_id).size()
    avg_len = tile_counts.mean()
    return avg_len


def get_nesting_stats(df, coarse_id_col, fine_id_col):
    """
    Calculates average number of fine tiles per coarse tile.
    """
    # Group by Coarse ID, then count unique Fine IDs within that group
    fine_per_coarse = df.groupby(coarse_id_col)[fine_id_col].nunique()
    return fine_per_coarse.mean()


def evaluate_domain(domain):
    print(f"\n{'=' * 20} EVALUATING DOMAIN: {domain.upper()} {'=' * 20}")

    human_dir = os.path.join(HUMAN_ROOT, domain)
    nn_dir = os.path.join(NN_ROOT, domain)

    # Storage for concatenation (to calculate global metrics)
    all_human_coarse = []
    all_nn_coarse = []
    all_human_fine = []
    all_nn_fine = []

    # Storage for descriptive stats (lists of averages per file)
    stats_human_coarse_len = []
    stats_sys_coarse_len = []
    stats_human_fine_len = []
    stats_sys_fine_len = []
    stats_human_nesting = []
    stats_sys_nesting = []

    files = glob.glob(os.path.join(human_dir, "*.tsv"))

    if not files:
        print("No files found.")
        return

    for human_path in files:
        filename = os.path.basename(human_path)
        nn_path = os.path.join(nn_dir, filename)

        if not os.path.exists(nn_path):
            print(f"Warning: Prediction file missing for {filename}")
            continue

        try:
            # Read Dataframes
            df_human = pd.read_csv(human_path, sep='\t')
            df_nn = pd.read_csv(nn_path, sep='\t')

            # Ensure lengths match (sanity check)
            min_len = min(len(df_human), len(df_nn))
            if len(df_human) != len(df_nn):
                print(f"Length mismatch in {filename}: GT={len(df_human)}, Sys={len(df_nn)}. Truncating to {min_len}.")
                df_human = df_human.iloc[:min_len]
                df_nn = df_nn.iloc[:min_len]

            # --- COLLECT BINARY LABELS FOR AGREEMENT ---
            all_human_coarse.extend(df_human[HUMAN_COLS['coarse_binary']].tolist())
            all_nn_coarse.extend(df_nn[NN_COLS['coarse_binary']].tolist())

            human_fine = df_human[HUMAN_COLS['fine_binary']].tolist()
            assert all([type(k) in [int, float] for k in human_fine])
            all_human_fine.extend(human_fine)
            # if len(all_human_fine) > 5366:
            #     k = 1
            #     print(k)
            all_nn_fine.extend(df_nn[NN_COLS['fine_binary']].tolist())

            # --- CALCULATE DESCRIPTIVE STATS ---
            # 1. Coarse Lengths
            stats_human_coarse_len.append(get_tile_stats(df_human, HUMAN_COLS['coarse_binary'], HUMAN_COLS['coarse_id']))
            stats_sys_coarse_len.append(get_tile_stats(df_nn, NN_COLS['coarse_binary'], NN_COLS['coarse_id']))

            # 2. Fine Lengths
            stats_human_fine_len.append(get_tile_stats(df_human, HUMAN_COLS['fine_binary'], HUMAN_COLS['fine_id']))
            stats_sys_fine_len.append(get_tile_stats(df_nn, NN_COLS['fine_binary'], NN_COLS['fine_id']))

            # 3. Nesting (Fine per Coarse)
            stats_human_nesting.append(get_nesting_stats(df_human, HUMAN_COLS['coarse_id'], HUMAN_COLS['fine_id']))
            stats_sys_nesting.append(get_nesting_stats(df_nn, NN_COLS['coarse_id'], NN_COLS['fine_id']))

        except KeyError as e:
            print(f"Column error in {filename}: {e}. Check your TSV headers.")
            continue

    # --- FINAL CALCULATIONS ---

    # 1. Agreement Metrics (Global)
    # We focus on the "positive" class (1 = New Tile)
    coarse_kappa = cohen_kappa_score(all_human_coarse, all_nn_coarse)
    coarse_f1 = f1_score(all_human_coarse, all_nn_coarse, pos_label=1)

    fine_kappa = cohen_kappa_score(all_human_fine, all_nn_fine)
    fine_f1 = f1_score(all_human_fine, all_nn_fine, pos_label=1)

    # 2. Descriptive Averages (Mean of means)
    print(f"\n--- DESCRIPTIVE STATISTICS (Avg Sentences) ---")
    print(f"{'Metric':<30} | {'Human (GT)':<12} | {'System':<12}")
    print("-" * 60)
    print(
        f"{'Avg Coarse Tile Length':<30} | {np.mean(stats_human_coarse_len):.2f}         | {np.mean(stats_sys_coarse_len):.2f}")
    print(
        f"{'Avg Fine Tile Length':<30} | {np.mean(stats_human_fine_len):.2f}         | {np.mean(stats_sys_fine_len):.2f}")
    print(
        f"{'Avg Fine Tiles per Coarse':<30} | {np.mean(stats_human_nesting):.2f}         | {np.mean(stats_sys_nesting):.2f}")

    print(f"\n--- AGREEMENT METRICS (Boundary Detection) ---")
    print(f"{'Level':<15} | {'Kappa':<10} | {'F1 (Class 1)':<10}")
    print("-" * 45)
    print(f"{'Coarse':<15} | {coarse_kappa:.3f}      | {coarse_f1:.3f}")
    print(f"{'Fine':<15} | {fine_kappa:.3f}      | {fine_f1:.3f}")


def main():
    for domain in SUBDIRS:
        if os.path.exists(os.path.join(HUMAN_ROOT, domain)):
            evaluate_domain(domain)
        else:
            print(f"Domain folder not found: {domain}")


if __name__ == "__main__":
    main()