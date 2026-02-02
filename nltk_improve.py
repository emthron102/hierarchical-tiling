import os
import csv
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.metrics.segmentation import windowdiff, pk
from sklearn.metrics import cohen_kappa_score

# DOWNLOAD ONCE
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")


TEXT_DIR = "data/vlogs"
DATA_DIR = "data/vlogs"
OUT_DIR = "results/improve/vlogs"

STOP = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_sentence(sent):
    return " ".join(
        lemmatizer.lemmatize(w.lower())
        for w in word_tokenize(sent)
        if w.isalpha() and w.lower() not in STOP
    )

def segments_to_boundaries(segments, sentences):
    y = np.zeros(len(sentences), dtype=int)
    idx = 0

    for seg in segments[:-1]:  # ignore last boundary
        seg_sents = sent_tokenize(seg)
        idx += len(seg_sents)
        if idx < len(y):
            y[idx] = 1
    return y


def cap_boundaries(y_pred, n_gold):
    idxs = np.where(y_pred == 1)[0]
    if len(idxs) <= n_gold:
        return y_pred

    capped = np.zeros_like(y_pred)
    capped[idxs[:n_gold]] = 1
    return capped


def window_k(y_gold):
    boundaries = np.where(y_gold == 1)[0]
    seg_lens = np.diff([0] + boundaries.tolist() + [len(y_gold)])
    return max(1, int(np.mean(seg_lens) / 2))


def eval_segmentation(y_gold, y_pred):
    k = window_k(y_gold)

    gold_str = "".join(map(str, y_gold))
    pred_str = "".join(map(str, y_pred))

    return {
        "kappa": cohen_kappa_score(y_gold, y_pred),
        "windowdiff": windowdiff(gold_str, pred_str, k),
        "pk": pk(gold_str, pred_str, k),
        "k": k,
        "gold_n": int(y_gold.sum()),
        "pred_n": int(y_pred.sum())
    }


def run_texttiling(text, sentences, y_gold):
    tt = TextTilingTokenizer(
        w=15,
        k=8,
        smoothing_width=3,
        smoothing_rounds=1,
        cutoff_policy="HC"
    )

    try:
        segments = tt.tokenize(text)
        y_pred = segments_to_boundaries(segments, sentences)
    except ValueError:
        y_pred = np.zeros(len(sentences), dtype=int)

    return cap_boundaries(y_pred, int(y_gold.sum()))


for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".tsv"):
        continue

    doc_id = fname.replace(".tsv", "")
    print(f"\nProcessing {doc_id}")

    # load human annotations
    tsv_path = os.path.join(DATA_DIR, fname)
    rows = []

    with open(tsv_path, encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    sentences = [r["sentence"] for r in rows]

    y_gold_coarse = np.array(
        [int(r["boundary-coarse"] or 0) for r in rows]
    )
    y_gold_fine = np.array(
        [int(r["boundary-fine"] or 0) for r in rows]
    )

    # load text
    # text_path = os.path.join(TEXT_DIR, f"{doc_id}.txt")
    # with open(text_path, encoding="utf8") as f:
    #     text = f.read()
    sentences = []
    tsv_path = os.path.join(DATA_DIR, f"{doc_id}.tsv")
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentences.append(row["sentence"])
    text = "\n\n".join(sentences)

    # predict
    y_pred_coarse = run_texttiling(text, sentences, y_gold_coarse)
    y_pred_fine   = run_texttiling(text, sentences, y_gold_fine)

    # metrics
    m_c = eval_segmentation(y_gold_coarse, y_pred_coarse)
    m_f = eval_segmentation(y_gold_fine, y_pred_fine)

    print(
        f"  COARSE κ={m_c['kappa']:.3f} WD={m_c['windowdiff']:.3f} PK={m_c['pk']:.3f} "
        f"(gold={m_c['gold_n']}, pred={m_c['pred_n']}, k={m_c['k']})"
    )
    print(
        f"  FINE   κ={m_f['kappa']:.3f} WD={m_f['windowdiff']:.3f} PK={m_f['pk']:.3f} "
        f"(gold={m_f['gold_n']}, pred={m_f['pred_n']}, k={m_f['k']})"
    )

    # write output TSV
    out_path = os.path.join(OUT_DIR, fname)

    with open(out_path, "w", encoding="utf8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow([
            "boundary-coarse",
            "segment-coarse",
            "boundary-fine",
            "segment-fine",
            "sentence"
        ])

        seg_c = 1
        seg_f = 1

        for i, sent in enumerate(sentences):
            if y_pred_coarse[i] == 1:
                seg_c += 1
            if y_pred_fine[i] == 1:
                seg_f += 1

            writer.writerow([
                int(y_pred_coarse[i]),
                seg_c,
                int(y_pred_fine[i]),
                seg_f,
                sent
            ])
