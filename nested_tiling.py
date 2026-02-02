import pandas as pd
from sentence_texttiling import TextTilingTokenizer

DELIMITER = "|<S>|"

class NestedTilingRunner:
    def __init__(self, use_tsv_boundaries=True, max_depth=2):
        self.use_tsv_boundaries = use_tsv_boundaries
        self.max_depth = max_depth

    def recursive_tokenize(self, text, current_depth, params_list):
        param_idx = min(current_depth - 1, len(params_list) - 1)
        current_params = params_list[param_idx]

        sep = DELIMITER if self.use_tsv_boundaries else None
        
        tt = TextTilingTokenizer(
            w=current_params.get('w', 20),
            k=current_params.get('k', 10),
            smoothing_width=current_params.get('smoothing_width', 2),
            cutoff_policy=current_params.get('cutoff_policy', 1),
            sentence_sep=sep
        )

        segments = tt.tokenize(text)

        results = []

        if len(segments) <= 1 or current_depth >= self.max_depth:
            for seg in segments:
                results.append({
                    "depth": current_depth,
                    "text": seg,
                    "children": []
                })
        else:
            for seg in segments:
                children = self.recursive_tokenize(seg, current_depth + 1, params_list)
                results.append({
                    "depth": current_depth,
                    "text": seg,
                    "children": children
                })
                
        return results

    def print_tree(self, nodes, indent=0):
        for i, node in enumerate(nodes):
            clean_text = node['text'].replace(DELIMITER, " ")
            preview = (clean_text[:60] + "...") if len(clean_text) > 60 else clean_text
            prefix = "    " * indent
            # starts from 1
            marker = f"L{node['depth']}-{i+1}"
            
            print(f"{prefix}[{marker}] {preview} (len: {len(clean_text)})")
            
            if node['children']:
                self.print_tree(node['children'], indent + 1)

    def flatten_tree_data(self, nodes, parent_indices=None):
        if parent_indices is None:
            parent_indices = {}

        all_rows = []

        for i, node in enumerate(nodes):
            current_depth = node['depth']
            current_indices = parent_indices.copy()
            current_indices[f"Layer_{current_depth}"] = i + 1

            if node['children']:
                children_rows = self.flatten_tree_data(node['children'], current_indices)
                all_rows.extend(children_rows)
            else:
                sentences = node['text'].split(DELIMITER)
                for sent in sentences:
                    if sent.strip():
                        row = {"sentence": sent}
                        row.update(current_indices)
                        all_rows.append(row)
        
        return all_rows

def main():
    target_file = "TextTiling_travel_vlog_data/travel_guides/berlin.tsv"
    output_filename = "nested_berlin.tsv"

    USE_TSV_BOUNDARIES = True
    MAX_DEPTH = 3
    
    PARAMS_PER_DEPTH = [
        {'w': 30, 'k': 10, 'cutoff_policy': 1, 'smoothing_width': 2}, 
        {'w': 10, 'k': 5,  'cutoff_policy': 1, 'smoothing_width': 2}, 
        {'w': 6,  'k': 3,  'cutoff_policy': 1, 'smoothing_width': 1}, 
    ]

    print(f"Loading {target_file}...")
    
    try:
        df = pd.read_csv(target_file, sep="\t")
        sentences = df['sentence'].tolist()

        if USE_TSV_BOUNDARIES:
            print("Mode: Using TSV provided boundaries (preserving original segmentation).")
            full_text = DELIMITER.join(sentences)
        else:
            print("Mode: Using Spacy sentence segmentation.")
            full_text = " ".join(sentences)

        runner = NestedTilingRunner(
            use_tsv_boundaries=USE_TSV_BOUNDARIES, 
            max_depth=MAX_DEPTH
        )
        
        print(f"Running Nested TextTiling (Max Depth: {MAX_DEPTH})...")
        tree_results = runner.recursive_tokenize(full_text, current_depth=1, params_list=PARAMS_PER_DEPTH)

        runner.print_tree(tree_results)
        
        flat_data = runner.flatten_tree_data(tree_results)
        df_output = pd.DataFrame(flat_data)

        layer_cols = sorted([c for c in df_output.columns if c.startswith("Layer_")])
        cols = ["sentence"] + layer_cols
        df_output = df_output[cols]
        df_output = df_output.fillna(0).astype({c: int for c in layer_cols})

        df_output.to_csv(output_filename, sep='\t', index=False)

        print(f"Saved detailed segmentation to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: File not found at {target_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()