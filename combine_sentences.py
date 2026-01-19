import pandas as pd

# Read the TSV
df = pd.read_csv("TextTiling_travel_vlog_data/vlogs/vlog10_beginning-gardening.tsv", sep="\t")

# Combine all sentences into one string
all_text = " ".join(df["sentence"].astype(str))

# Write to a text file
with open("data/vlogs/beginning_gardening.txt", "w", encoding="utf-8") as f:
    f.write(all_text)