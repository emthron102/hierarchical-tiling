import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# --- Configuration ---
INPUT_ROOT = './data'
OUTPUT_ROOT = './results'
SUBDIRS = ['travel_guides', 'vlogs']

# Adjust these thresholds based on your SBERT model and data sensitivity
COARSE_THRESHOLD = 0.90
FINE_THRESHOLD = 0.70

MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast and effective


def process_file(input_path, output_path, model):
    """
    Reads a TSV, computes tiles, and writes the 5-column output TSV.
    """
    print(f"Processing: {input_path}")

    # 1. Read Data
    # Assuming the file has headers. If not, remove 'header=0' and use 'names=['...', 'sentence']'
    try:
        df = pd.read_csv(input_path, sep='\t')

        # Verify 'sentence' column exists
        if 'sentence' not in df.columns:
            # Fallback: If no header, assume last column is sentence
            # df = pd.read_csv(input_path, sep='\t', header=None)
            # sentences = df.iloc[:, -1].tolist()
            print(f"Skipping {input_path}: Column 'sentence' not found.")
            return

        sentences = df['sentence'].astype(str).tolist()

    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return

    if not sentences:
        print(f"Skipping {input_path}: Empty file.")
        return

    # 2. SBERT Encoding & Distance Calculation
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Calculate cosine distances between sentence i and i+1
    # Result is an array of size N-1
    if len(sentences) > 1:
        dists = np.diagonal(cosine_distances(embeddings[:-1], embeddings[1:]))
    else:
        dists = []

    # 3. Generate Tiling Columns
    # We initialize lists to store the column data
    binary_new_coarse = []
    coarse_ids = []
    binary_new_fine = []
    fine_ids = []

    # State variables
    current_coarse_id = 0
    current_fine_id = 0

    # --- HANDLE FIRST SENTENCE ---
    # The first sentence always starts the first tile
    binary_new_coarse.append(1)
    coarse_ids.append(current_coarse_id)
    binary_new_fine.append(1)
    fine_ids.append(current_fine_id)

    # --- HANDLE SUBSEQUENT SENTENCES ---
    for i, dist in enumerate(dists):
        # i corresponds to the gap between sentence[i] and sentence[i+1]
        # We are deciding the status of sentence[i+1]

        is_new_coarse = 0
        is_new_fine = 0

        if dist >= COARSE_THRESHOLD:
            # New Coarse implies New Fine automatically
            current_coarse_id += 1
            current_fine_id += 1
            is_new_coarse = 1
            is_new_fine = 1

        elif dist >= FINE_THRESHOLD:
            # New Fine only
            current_fine_id += 1
            is_new_fine = 1

        else:
            # Continue current tiles
            pass

        binary_new_coarse.append(is_new_coarse)
        coarse_ids.append(current_coarse_id)
        binary_new_fine.append(is_new_fine)
        fine_ids.append(current_fine_id)

    # 4. Create Result DataFrame
    result_df = pd.DataFrame({
        'binary_prediction_new_coarse_tile': binary_new_coarse,
        'coarse_tile_number': coarse_ids,
        'binary_prediction_new_fine_tile': binary_new_fine,
        'fine_tile_number': fine_ids,
        'sentence': sentences
    })

    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, sep='\t', index=False)


def main():
    # Load model once
    print(f"Loading SBERT model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    for subdir in SUBDIRS:
        read_dir = os.path.join(INPUT_ROOT, subdir)
        write_dir = os.path.join(OUTPUT_ROOT, subdir)

        # Check if source directory exists
        if not os.path.exists(read_dir):
            print(f"Directory not found: {read_dir}")
            continue

        # Get all .tsv files
        files = glob.glob(os.path.join(read_dir, "*.tsv"))
        print(f"Found {len(files)} TSV files in {subdir}")

        for file_path in files:
            # Construct output filename
            filename = os.path.basename(file_path)
            output_path = os.path.join(write_dir, filename)

            process_file(file_path, output_path, model)

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()