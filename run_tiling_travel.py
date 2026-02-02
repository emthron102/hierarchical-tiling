
import os
import csv
from nltk.tokenize import sent_tokenize, TextTilingTokenizer
from sklearn.metrics import cohen_kappa_score
from nltk.metrics import windowdiff

TEXT_DIR = "data/travel_guides"
DATA_DIR = "TextTiling_travel_vlog_data/travel_guides"
#OUT_DIR = "results/wiki_grid"
#OUT_DIR = "results/human_grid"
OUT_DIR = "results/baseline"

os.makedirs(OUT_DIR, exist_ok=True)

# Chosen parameters from prior grid search
# from wiki-based grid search:
# COARSE_PARAMS = dict(w=20, k=12, smoothing_width=3, smoothing_rounds=2, cutoff_policy="HC")
# FINE_PARAMS = dict(w=20, k=4, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC")

# from human annotation grid search:
COARSE_PARAMS = dict(w=50, k=25, smoothing_width=5, smoothing_rounds=3, cutoff_policy="HC")
FINE_PARAMS = dict(w=35, k=15, smoothing_width=5, smoothing_rounds=3, cutoff_policy="HC")


# helpers
def load_human_boundaries(doc_filename, granularity="coarse"):
    sentences = []
    y_gold = []
    tsv_path = os.path.join(DATA_DIR, f"{doc_filename}.tsv")
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        boundary_col = "boundary-coarse" if granularity == "coarse" else "boundary-fine"
        for row in reader:
            sentences.append(row["sentence"])
            y_gold.append(int(row[boundary_col]) if row[boundary_col] else 0)
    return sentences, y_gold

def segments_to_breaks(segments):
    sent_count = 0
    breaks = []
    for seg in segments:
        seg_sents = sent_tokenize(seg)
        sent_count += len(seg_sents)
        breaks.append(sent_count)
    return breaks[:-1]  # ignore end of doc boundary

def breaks_to_labels(breaks, n_sentences):
    labels = [0] * (n_sentences)
    for b in breaks:
        if b-1 < len(labels):
            labels[b - 1] = 1
    return labels

def assign_segments(labels):
    segments = []
    seg_id = 1
    for l in labels:
        segments.append(seg_id)
        if l == 1:
            seg_id += 1
    segments.append(seg_id)  # last sentence
    return segments

# run text tiling
doc_ids = [f[:-4] for f in os.listdir(TEXT_DIR) if f.endswith(".txt")]

for doc_id in doc_ids:
    print(f"\nTiling: {doc_id}")
    # Load text
    with open(os.path.join(TEXT_DIR, f"{doc_id}.txt"), "r", encoding="utf-8") as f:
        doc_text = f.read()

    # Load human boundaries
    sentences, y_gold_coarse = load_human_boundaries(doc_id, "coarse")
    _, y_gold_fine = load_human_boundaries(doc_id, "fine")

    # Coarse segmentation
    tt_coarse = TextTilingTokenizer(**COARSE_PARAMS)
    tt_coarse = TextTilingTokenizer() # baseline
    coarse_segments = tt_coarse.tokenize(doc_text)
    coarse_breaks = segments_to_breaks(coarse_segments)
    y_pred_coarse = breaks_to_labels(coarse_breaks, len(sentences))
    coarse_seg_ids = assign_segments(y_pred_coarse)
    kappa_coarse = cohen_kappa_score(y_gold_coarse, y_pred_coarse)
    print(f"  Coarse κ={kappa_coarse:.3f}")

    # Fine segmentation
    #tt_fine = TextTilingTokenizer(**FINE_PARAMS)
    tt_fine = TextTilingTokenizer() # baseline
    fine_segments = tt_fine.tokenize(doc_text)
    fine_breaks = segments_to_breaks(fine_segments)
    y_pred_fine = breaks_to_labels(fine_breaks, len(sentences))
    fine_seg_ids = assign_segments(y_pred_fine)
    kappa_fine = cohen_kappa_score(y_gold_fine, y_pred_fine)
    print(f"  Fine κ={kappa_fine:.3f}")

    # Write TSV
    out_tsv = os.path.join(OUT_DIR, f"{doc_id}.tsv")
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["boundary-coarse","segment-coarse","boundary-fine","segment-fine","sentence"])
        for i, sent in enumerate(sentences):
            b_coarse = y_pred_coarse[i] if i < len(y_pred_coarse) else 0
            s_coarse = coarse_seg_ids[i]
            b_fine = y_pred_fine[i] if i < len(y_pred_fine) else 0
            s_fine = fine_seg_ids[i]
            writer.writerow([b_coarse, s_coarse, b_fine, s_fine, sent])

    print(f"Saved to {out_tsv}")
