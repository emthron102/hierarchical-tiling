import json
import os
from nltk.tokenize import sent_tokenize

# get gold paragraph breaks
def extract_gold_from_doc(doc_text):
    """
    Given raw document text with paragraph breaks,
    return (sentences, gold_breaks)
    """

    # Split into paragraphs
    paragraphs = [p.strip() for p in doc_text.split('\n\n') if p.strip()]

    sentences = []
    gold_breaks = []
    sent_count = 0

    print(len(paragraphs), "paragraphs found.")

    for para in paragraphs:
        para_sents = sent_tokenize(para)
        sentences.extend(para_sents)
        sent_count += len(para_sents)
        gold_breaks.append(sent_count)

    # Remove final document boundary -- just end of doc
    gold_breaks = gold_breaks[:-1]

    return sentences, gold_breaks

DATA_DIR = "data/travel_guides"
output = {}

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".txt"):
        continue

    doc_id = os.path.splitext(fname)[0]

    with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
        doc_text = f.read()

    sentences, gold_breaks = extract_gold_from_doc(doc_text)

    output[doc_id] = {
        "sentences": sentences,
        "gold_breaks": gold_breaks
    }

    print(f"{doc_id}: {len(sentences)} sentences, {len(gold_breaks)} gold breaks")


with open("gold_paragraph_breaks.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

# sanity check -- cross check w the txt doc
doc = output["auckland"]
for b in doc["gold_breaks"]:
    print("BREAK AFTER:", doc["sentences"][b-1])
    print("NEXT SENT :", doc["sentences"][b])
    print()