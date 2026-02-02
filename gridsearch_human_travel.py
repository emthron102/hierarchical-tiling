'''
Grid search for TextTiling parameters on travel guide data
Evaluates against HUMAN/STUDENT ANNOTATED SEGMENT boundaries
'''

import os
import csv
import json
from nltk.metrics import windowdiff, pk
from nltk.tokenize import TextTilingTokenizer, sent_tokenize
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

# parameter sets
# COARSE_PARAMS = [
#     dict(w=30, k=15, smoothing_width=5, smoothing_rounds=2, cutoff_policy="HC"),
#     dict(w=40, k=20, smoothing_width=5, smoothing_rounds=2, cutoff_policy="HC"),
#     dict(w=50, k=25, smoothing_width=5, smoothing_rounds=3, cutoff_policy="HC"),
# ]

# COARSE_PARAMS = [
#     dict(w=50, k=30, smoothing_width=5, smoothing_rounds=3, cutoff_policy="HC"),
#     dict(w=60, k=35, smoothing_width=7, smoothing_rounds=3, cutoff_policy="HC"),
#     dict(w=70, k=40, smoothing_width=7, smoothing_rounds=4, cutoff_policy="HC"),
# ]

# COARSE_PARAMS = [
#     dict(w=20, k=15, smoothing_width=3, smoothing_rounds=2, cutoff_policy="HC"),
#     dict(w=20, k=25, smoothing_width=5, smoothing_rounds=2, cutoff_policy="HC"),
#     dict(w=40, k=30, smoothing_width=5, smoothing_rounds=3, cutoff_policy="HC"),
# ]

COARSE_PARAMS = [
    dict(w=60, k=30, smoothing_width=7, smoothing_rounds=3, cutoff_policy="HC"),
    dict(w=80, k=40, smoothing_width=9, smoothing_rounds=3, cutoff_policy="HC"),
    dict(w=100, k=50, smoothing_width=11, smoothing_rounds=4, cutoff_policy="HC"),
]

# FINE_PARAMS = [
#     dict(w=10, k=3, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
#     dict(w=15, k=4, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
#     dict(w=20, k=5, smoothing_width=2, smoothing_rounds=1, cutoff_policy="LC"),
# ]

# FINE_PARAMS = [
#     dict(w=5, k=2, smoothing_width=1, smoothing_rounds=0, cutoff_policy="LC"),
#     dict(w=8, k=3, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
#     dict(w=10, k=4, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
# ]

# FINE_PARAMS = [
#     dict(w=20, k=4, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
#     dict(w=20, k=6, smoothing_width=2, smoothing_rounds=1, cutoff_policy="LC"),
#     dict(w=10, k=8, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
# ]

FINE_PARAMS = [
    dict(w=25, k=10, smoothing_width=3, smoothing_rounds=2, cutoff_policy="HC"),
    dict(w=30, k=12, smoothing_width=5, smoothing_rounds=2, cutoff_policy="HC"),
    dict(w=35, k=15, smoothing_width=5, smoothing_rounds=3, cutoff_policy="HC"),
]

ALL_PARAMS = (
    [("coarse", p) for p in COARSE_PARAMS]
    + [("fine", p) for p in FINE_PARAMS]
)

TEXT_DIR = "data/travel_guides"
DATA_DIR = "TextTiling_travel_vlog_data/travel_guides"
OUT_JSON = "last_travel_grid_human.json"

doc_ids = [filename[:-4] for filename in os.listdir(TEXT_DIR)]

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


# grid search

rows = []
param_results = []
best_by_granularity = {
    "coarse": {"avg_kappa": -1},
    "fine": {"avg_kappa": -1},
}
# best_by_granularity = {
#     "coarse": {"windowdiff": float("inf")}, 
#     "fine": {"windowdiff": float("inf")}
# }


total_runs = len(ALL_PARAMS) * len(doc_ids)
run_idx = 0

for granularity, params in ALL_PARAMS:
    print(f"Testing {granularity.upper()} parameters:")
    print(params)

    tt = TextTilingTokenizer(**params)
    kappas = []
    windowdiffs = []
    pks = []
    f1s = []

    for doc_i, doc_id in enumerate(doc_ids, start=1):
        run_idx += 1
        print(
            f"[{run_idx}/{total_runs}] "
            f"{granularity} | doc {doc_i}/{len(doc_ids)} | {doc_id}"
        )

        # get gold labels from the human annotated data
        sentences, y_gold = load_human_boundaries(doc_id, granularity)

        with open(os.path.join(TEXT_DIR, f"{doc_id}.txt"), "r", encoding="utf-8") as f:
            doc_text = f.read()

        segments = tt.tokenize(doc_text)
        pred_breaks = segments_to_breaks(segments)
        y_pred = breaks_to_labels(pred_breaks, len(sentences))

        kappa = cohen_kappa_score(y_gold, y_pred)

        # recommended to use half the average segment length in sentences
        num_segments = sum(y_gold) + 1 
        n_sentences = len(sentences)
        avg_seg_len = n_sentences / num_segments
        wd_k = max(1, int(avg_seg_len / 2))
        wd = windowdiff(y_gold, y_pred, k=wd_k)

        print(f"y_gold len={len(y_gold)}, y_pred len={len(y_pred)}")
        print(f"y_gold={y_gold}")
        print(f"y_pred={y_pred}")
        print(f"WD k={wd_k}")

        pk_score = pk(y_gold, y_pred, k=wd_k)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_gold, y_pred, average="binary", zero_division=0
        )

        kappas.append(kappa)
        windowdiffs.append(wd)
        pks.append(pk_score)
        f1s.append(f1)

        rows.append({
            "doc_id": doc_id,
            "granularity": granularity,
            **params,
            "kappa": kappa,
            "windowdiff": wd,
            "pk": pk_score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_gold_boundaries": sum(y_gold),
            "num_pred_boundaries": sum(y_pred),
        })

        print(
            f"    κ={kappa:.3f} | F1={f1:.3f} | WindowDiff={wd:.3f} | PK={pk_score:.3f}"
            f"(gold={sum(y_gold)}, pred={sum(y_pred)})"
        )

    avg_kappa = sum(kappas) / len(kappas)
    avg_f1 = sum(f1s) / len(f1s)
    avg_windowdiff = sum(windowdiffs) / len(windowdiffs)
    avg_pk = sum(pks) / len(pks)

    print("-" * 60)
    print(
        f"Average for {granularity.upper()} params:"
        f" κ={avg_kappa:.3f}, F1={avg_f1:.3f}, WindowDiff={avg_windowdiff:.3f}, PK={avg_pk:.3f}"
    )

    param_summary = {
        "granularity": granularity,
        "params": params,
        "avg_kappa": avg_kappa,
        "avg_f1": avg_f1,
        "avg_windowdiff": avg_windowdiff,
        "avg_pk": avg_pk,
    }
    param_results.append(param_summary)

    if avg_kappa > best_by_granularity[granularity]["avg_kappa"]:
        best_by_granularity[granularity] = param_summary
        print("New best for ", granularity, ": ", best_by_granularity[granularity])
    # if avg_windowdiff < best_by_granularity[granularity]["avg_windowdiff"]:
    #     best_by_granularity[granularity] = param_summary
    #     print(f"New best for {granularity}: {param_summary}")

# save all param results
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "all_parameter_results": param_results,
            "best_by_granularity": best_by_granularity,
        },
        f,
        indent=2,
    )

print("\nFinal results")
for g, info in best_by_granularity.items():
    print(f"\n{g.upper()}:")
    print(f"Params: {info['params']}")
    print(f"Avg κ: {info['avg_kappa']:.3f}")
    print(f"Avg F1: {info['avg_f1']:.3f}")
    print(f"Avg WindowDiff: {info['avg_windowdiff']:.3f}")
    print(f"Avg PK: {info['avg_pk']:.3f}")
print(f"Saved grid summary results to {OUT_JSON}")
