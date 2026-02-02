'''
Grid search for TextTiling parameters on travel guide data
Evaluates against WIKI VOYAGE paragraph boundaries
'''

import csv
import json
import os
from nltk.metrics.segmentation import windowdiff
from nltk.tokenize import TextTilingTokenizer, sent_tokenize
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score


# "gold" paragraph breaks -- from wikivoyage
with open("gold_paragraph_breaks.json", "r", encoding="utf-8") as f:
    gold_data = json.load(f)

doc_ids = list(gold_data.keys())

# parameter sets
COARSE_PARAMS = [
    dict(w=20, k=12, smoothing_width=3, smoothing_rounds=2, cutoff_policy="HC"),
    dict(w=20, k=15, smoothing_width=5, smoothing_rounds=2, cutoff_policy="HC"),
    dict(w=40, k=20, smoothing_width=5, smoothing_rounds=3, cutoff_policy="HC"),
]

FINE_PARAMS = [
    dict(w=20, k=4, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
    dict(w=20, k=6, smoothing_width=2, smoothing_rounds=1, cutoff_policy="LC"),
    dict(w=10, k=5, smoothing_width=1, smoothing_rounds=1, cutoff_policy="LC"),
]

ALL_PARAMS = (
    [("coarse", p) for p in COARSE_PARAMS]
    + [("fine", p) for p in FINE_PARAMS]
)

TEXT_DIR = "data/travel_guides"
OUT_JSON = "travel_grid_summary.json"

# helpers
def segments_to_breaks(segments):
    sent_count = 0
    breaks = []
    for seg in segments:
        seg_sents = sent_tokenize(seg)
        sent_count += len(seg_sents)
        breaks.append(sent_count)
    return breaks[:-1]  # ignore end of doc boundary

def breaks_to_labels(breaks, n_sentences):
    labels = [0] * (n_sentences - 1)
    for b in breaks:
        if 0 < b <= n_sentences - 1:
            labels[b - 1] = 1
    return labels


# grid search
rows = []
param_results = []
best_by_granularity = {
    "coarse": {"avg_kappa": -1},
    "fine": {"avg_kappa": -1},
}

total_runs = len(ALL_PARAMS) * len(doc_ids)
run_idx = 0

for granularity, params in ALL_PARAMS:
    print(f"Testing {granularity.upper()} parameters:")
    print(params)

    tt = TextTilingTokenizer(**params)
    kappas = []
    f1s = []

    for doc_i, doc_id in enumerate(doc_ids, start=1):
        run_idx += 1
        print(
            f"[{run_idx}/{total_runs}] "
            f"{granularity} | doc {doc_i}/{len(doc_ids)} | {doc_id}"
        )

        doc_info = gold_data[doc_id]
        sentences = doc_info["sentences"]
        gold_breaks = doc_info["gold_breaks"]

        with open(os.path.join(TEXT_DIR, f"{doc_id}.txt"), "r", encoding="utf-8") as f:
            doc_text = f.read()

        segments = tt.tokenize(doc_text)
        pred_breaks = segments_to_breaks(segments)

        y_gold = breaks_to_labels(gold_breaks, len(sentences))
        y_pred = breaks_to_labels(pred_breaks, len(sentences))

        kappa = cohen_kappa_score(y_gold, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_gold, y_pred, average="binary", zero_division=0
        )

        kappas.append(kappa)
        f1s.append(f1)

        rows.append({
            "doc_id": doc_id,
            "granularity": granularity,
            **params,
            "kappa": kappa,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_gold_boundaries": sum(y_gold),
            "num_pred_boundaries": sum(y_pred),
        })

        print(
            f"    κ={kappa:.3f} | F1={f1:.3f} "
            f"(gold={sum(y_gold)}, pred={sum(y_pred)})"
        )

    avg_kappa = sum(kappas) / len(kappas)
    avg_f1 = sum(f1s) / len(f1s)

    print("-" * 60)
    print(
        f"Average for {granularity.upper()} params:"
        f"κ={avg_kappa:.3f}, F1={avg_f1:.3f}"
    )

    param_summary = {
        "granularity": granularity,
        "params": params,
        "avg_kappa": avg_kappa,
        "avg_f1": avg_f1,
    }
    param_results.append(param_summary)

    if avg_kappa > best_by_granularity[granularity]["avg_kappa"]:
        best_by_granularity[granularity] = param_summary
        print("New best for ",granularity,": ", best_by_granularity[granularity])

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

print(f"Saved grid summary results to {OUT_JSON}")


# for w in w_values:
#     for k in k_values:
#         # new tiler with these params
#         tt = TextTilingTokenizer(w=w, k=k)
#         f1_scores = []

#         print(f"\nTesting w={w}, k={k} ...")
#         for i, doc_id in enumerate(train_ids, start=1):
#             print(f"  Train doc {i}/{len(train_ids)}: {doc_id}")
#             doc_info = gold_data[doc_id]
#             sentences = doc_info["sentences"]
#             gold_breaks = doc_info["gold_breaks"]

#             # open text file
#             txt_path = os.path.join("data/travel_guides", f"{doc_id}.txt")
#             with open(txt_path, "r", encoding="utf-8") as f:
#                 doc_text = f.read()

#             segments = tt.tokenize(doc_text)

#             # convert segments to sentence-level breaks for comparison
#             predicted_breaks = []
#             sent_count = 0
#             for seg in segments:
#                 seg_sents = sent_tokenize(seg)
#                 sent_count += len(seg_sents)
#                 predicted_breaks.append(sent_count)
#             predicted_breaks = predicted_breaks[:-1]

#             # make binary
#             y_gold = breaks_to_labels(gold_breaks, len(sentences))
#             y_pred = breaks_to_labels(predicted_breaks, len(sentences))

#             # metrics
#             f1 = precision_recall_fscore_support(y_gold, y_pred, average="binary")[2]
#             f1_scores.append(f1)

#             print(f"    F1: {f1:.3f}")

#         avg_f1 = sum(f1_scores)/len(f1_scores)
#         print(f"  Avg train F1 for w={w}, k={k}: {avg_f1:.3f}")

#         if avg_f1 > best_f1:
#             best_f1 = avg_f1
#             best_params = (w, k)

# print(f"\nBest train params: w={best_params[0]}, k={best_params[1]} with F1={best_f1:.3f}")

# # evaluate
# print("\nEvaluating best params on eval docs...")
# tt_best = TextTilingTokenizer(w=best_params[0], k=best_params[1])
# eval_f1_scores = []

# for i, doc_id in enumerate(eval_ids, start=1):
#     print(f"  Eval doc {i}/{len(eval_ids)}: {doc_id}")
#     doc_info = gold_data[doc_id]
#     sentences = doc_info["sentences"]
#     gold_breaks = doc_info["gold_breaks"]

#     txt_path = os.path.join("data/travel_guides", f"{doc_id}.txt")
#     with open(txt_path, "r", encoding="utf-8") as f:
#         doc_text = f.read()

#     segments = tt_best.tokenize(doc_text)

#     predicted_breaks = []
#     sent_count = 0
#     for seg in segments:
#         seg_sents = sent_tokenize(seg)
#         sent_count += len(seg_sents)
#         predicted_breaks.append(sent_count)
#     predicted_breaks = predicted_breaks[:-1]

#     y_gold = breaks_to_labels(gold_breaks, len(sentences))
#     y_pred = breaks_to_labels(predicted_breaks, len(sentences))

#     f1 = precision_recall_fscore_support(y_gold, y_pred, average="binary")[2]
#     eval_f1_scores.append(f1)
#     print(f"    F1: {f1:.3f}")

# avg_eval_f1 = sum(eval_f1_scores)/len(eval_f1_scores)
# print(f"\nAverage eval F1 with best params: {avg_eval_f1:.3f}")
