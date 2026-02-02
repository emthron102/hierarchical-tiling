import math
import re
from nltk.corpus import stopwords as sw
import numpy
import spacy
import csv

BLOCK_COMPARISON = 0
LC, HC = 0, 1
DEFAULT_SMOOTHING = [0]


class TextTilingTokenizer:
    """
    TextTiling tokenizer for sentence-level segmentation.
    
    Uses spacy for sentence detection and normalizes topic boundaries to the nearest sentence break based on token counts.
    """

    def __init__(
        self,
        w: int = 20,
        k: int = 10,
        similarity_method: int = BLOCK_COMPARISON,
        smoothing_method: list[int] = DEFAULT_SMOOTHING,
        smoothing_width: int = 2,
        cutoff_policy: int = HC
    ):
        stopwords = sw.words("english")
        self.stopwords = set(stopwords)

        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        self.w = w
        self.k = k
        self.similarity_method = similarity_method
        self.smoothing_method = smoothing_method
        self.smoothing_width = smoothing_width
        self.cutoff_policy = cutoff_policy

    def _sentence_tokenize(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def tokenize(self, text: str) -> list[str]:
        segs = self.tokenize_with_spans(text)
        return [d["text"] for d in segs]

    def tokenize_with_spans(self, text: str) -> list[dict]:
        """Return sentence-level segments WITH sentence-span metadata.

        Each returned dict has:
          - segment_id: 0-based segment index
          - start_sentence: inclusive sentence index
          - end_sentence: exclusive sentence index
          - text: the segment text
        """
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [{"segment_id": 0, "start_sentence": 0, "end_sentence": 1, "text": sentences[0]}]

        # Count tokens per sentence in the same token space as `_divide_to_tokensequences`:
        # lowercase + nopunct filter + `[a-z]+` token regex.
        sent_token_counts: list[int] = []
        for sent in sentences:
            sent_lc = sent.lower()
            sent_nopunct = "".join(c for c in sent_lc if re.match(r"[a-z\-' \n\t]", c))
            sent_tokens = re.findall(r"[a-z]+", sent_nopunct)
            sent_token_counts.append(len(sent_tokens))

        sentence_boundaries_in_tokens = numpy.cumsum(sent_token_counts)

        full_text_for_processing = "\n\n".join(sentences)

        lowercase_text = full_text_for_processing.lower()
        nopunct_text = "".join(c for c in lowercase_text if re.match(r"[a-z\-' \n\t]", c))

        tokseqs = self._divide_to_tokensequences(nopunct_text)

        if len(tokseqs) < 2:
            return [{"segment_id": 0, "start_sentence": 0, "end_sentence": len(sentences), "text": text}]

        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list if wi[0] not in self.stopwords]

        token_table = self._create_token_table(tokseqs)

        gap_scores = self._block_comparison(tokseqs, token_table)

        if len(gap_scores) < 3:
            return [{"segment_id": 0, "start_sentence": 0, "end_sentence": len(sentences), "text": text}]

        smooth_scores = self._smooth_scores(gap_scores)
        depth_scores = self._depth_scores(smooth_scores)
        segment_boundaries_binary = self._identify_boundaries(depth_scores)

        final_sentence_indices = [0]

        for i, is_boundary in enumerate(segment_boundaries_binary):
            if is_boundary:
                boundary_token_idx = (i + 1) * self.w
                closest_sent_idx = numpy.abs(sentence_boundaries_in_tokens - boundary_token_idx).argmin()

                if closest_sent_idx + 1 not in final_sentence_indices and closest_sent_idx < len(sentences) - 1:
                    final_sentence_indices.append(closest_sent_idx + 1)

        final_sentence_indices.append(len(sentences))
        final_sentence_indices.sort()

        segments: list[dict] = []
        seg_id = 0
        for start, end in zip(final_sentence_indices[:-1], final_sentence_indices[1:]):
            segment_sentences = sentences[start:end]
            segments.append({
                "segment_id": seg_id,
                "start_sentence": start,
                "end_sentence": end,
                "text": " ".join(segment_sentences),
            })
            seg_id += 1

        return segments

    def _divide_to_tokensequences(self, text: str) -> list['TokenSequence']:
        w = self.w
        wrdindex_list = []
        # Use [a-z]+ to match only alphabetic tokens (text is already lowercased)
        for match in re.finditer(r"[a-z]+", text):
            wrdindex_list.append((match.group(), match.start()))
        
        return [
            TokenSequence(i // w, wrdindex_list[i:i + w])
            for i in range(0, len(wrdindex_list), w)
        ]

    def _create_token_table(self, token_sequences: list['TokenSequence']) -> dict[str, 'TokenTableField']:
        token_table = {}
        current_tok_seq = 0
        
        for ts in token_sequences:
            for word, index in ts.wrdindex_list:
                if word not in token_table:
                    token_table[word] = TokenTableField(
                        first_pos=index,
                        ts_occurences=[[current_tok_seq, 1]],
                        total_count=1,
                        last_tok_seq=current_tok_seq,
                    )
                else:
                    entry = token_table[word]
                    entry.total_count += 1
                    
                    if entry.last_tok_seq != current_tok_seq:
                        entry.last_tok_seq = current_tok_seq
                        entry.ts_occurences.append([current_tok_seq, 1])
                    else:
                        entry.ts_occurences[-1][1] += 1
            current_tok_seq += 1
        return token_table

    def _block_comparison_with_lag(
        self,
        tokseqs: list['TokenSequence'],
        token_table: dict[str, 'TokenTableField'],
        lag: int = 2,
    ) -> list[float]:
        """Compute cosine similarity between token-frequency blocks separated by a fixed lag.

        This is analogous to `_block_comparison`, but instead of comparing blocks around a single gap
        (curr_gap vs curr_gap+1), it compares a block ending at position `i` with a block starting at
        position `i + lag`.

        Returns a list of similarity scores of length `len(tokseqs) - lag`.
        """
        if lag < 1:
            raise ValueError("lag must be >= 1")
        if len(tokseqs) <= lag:
            return []

        def blk_frq(tok: str, block_set: set[int]) -> int:
            count = 0
            for seq_id, freq in token_table[tok].ts_occurences:
                if seq_id in block_set:
                    count += freq
            return count

        scores: list[float] = []
        n = len(tokseqs)

        # We compare windows of up to k token-sequences ending at i and starting at i+lag.
        # For each i, use the largest window that fits in-bounds for both blocks.
        for i in range(0, n - lag):
            # how many token-sequences can we look back from i (inclusive)
            left_cap = i + 1
            # how many token-sequences are available starting from i+lag (inclusive)
            right_cap = n - (i + lag)

            # window must fit for both left look-back and right look-ahead
            window_size = min(self.k, left_cap, right_cap)
            if window_size <= 0:
                continue

            # Left block: window_size sequences ending at i
            b1_set = {ts.index for ts in tokseqs[i - window_size + 1 : i + 1]}
            # Right block: window_size sequences starting at i+lag
            b2_set = {ts.index for ts in tokseqs[i + lag : i + lag + window_size]}

            score_dividend, score_divisor_b1, score_divisor_b2 = 0.0, 0.0, 0.0
            for t in token_table:
                freq1 = blk_frq(t, b1_set)
                freq2 = blk_frq(t, b2_set)

                score_dividend += freq1 * freq2
                score_divisor_b1 += freq1 ** 2
                score_divisor_b2 += freq2 ** 2

            if score_divisor_b1 > 0 and score_divisor_b2 > 0:
                score = score_dividend / math.sqrt(score_divisor_b1 * score_divisor_b2)
            else:
                score = 0.0
            scores.append(score)

        return scores

    def find_nesting(
        self,
        text: str,
        *,
        lag: int = 2,
        sim_quantile: float = 0.7,
        depth_quantile: float = 0.7,
        min_gap_tokseq: int = 1,
        min_gap_sentences: int = 1,
    ) -> list[tuple[int, int, int, int]]:
        """Heuristic detector for *nested* topic boundaries.

        We look for indices i such that:
          1) block similarity between i and i+lag (via `_block_comparison_with_lag`) is high
          2) left_depth score at gap i is high
          3) right_depth score at gap (i+lag-1) is high

        Returns a list of (left_start, left_end, right_start, right_end) sentence-boundary indices (0..len(sentences)), where left/right correspond to the compared blocks.
        """
        sentences = self._sentence_tokenize(text)
        if len(sentences) < 2:
            return []

        if min_gap_tokseq < 0 or min_gap_sentences < 0:
            raise ValueError("min_gap_tokseq and min_gap_sentences must be >= 0")
        # In tokseq space, left ends at i and right starts at i+lag, so the number of tokseqs in-between is (lag-1).
        if lag < (min_gap_tokseq + 1):
            raise ValueError(
                f"lag={lag} is too small for min_gap_tokseq={min_gap_tokseq}; need lag >= {min_gap_tokseq + 1}"
            )

        # Count tokens per sentence in the same token space as `_divide_to_tokensequences`:
        # lowercase + nopunct filter + `[a-z]+` token regex.
        sent_token_counts: list[int] = []
        for sent in sentences:
            sent_lc = sent.lower()
            sent_nopunct = "".join(c for c in sent_lc if re.match(r"[a-z\-' \n\t]", c))
            sent_tokens = re.findall(r"[a-z]+", sent_nopunct)
            sent_token_counts.append(len(sent_tokens))
        sentence_boundaries_in_tokens = numpy.cumsum(sent_token_counts)

        full_text_for_processing = "\n\n".join(sentences)
        lowercase_text = full_text_for_processing.lower()
        nopunct_text = "".join(c for c in lowercase_text if re.match(r"[a-z\-' \n\t]", c))

        tokseqs = self._divide_to_tokensequences(nopunct_text)
        if len(tokseqs) < 3:
            return []

        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list if wi[0] not in self.stopwords]

        token_table = self._create_token_table(tokseqs)

        # Standard TextTiling signals
        gap_scores = self._block_comparison(tokseqs, token_table)
        if len(gap_scores) < 3:
            return []
        smooth_scores = self._smooth_scores(gap_scores)
        depth_scores = self._depth_scores(smooth_scores)

        # Lagged similarity
        lag_scores = self._block_comparison_with_lag(tokseqs, token_table, lag=lag)
        if not lag_scores:
            return []

        # Thresholds via quantiles
        sim_q = float(numpy.clip(sim_quantile, 0.0, 1.0))
        dep_q = float(numpy.clip(depth_quantile, 0.0, 1.0))
        sim_thresh = float(numpy.quantile(numpy.array(lag_scores), sim_q))
        dep_thresh = float(numpy.quantile(numpy.array(depth_scores), dep_q))

        results: list[tuple[int, int, int, int]] = []
        def tokpos_to_sent_boundary(tok_pos: int) -> int:
            """Map a token-position (in the same token space as tokseqs) to a sentence boundary index."""
            if tok_pos <= 0:
                return 0
            idx = int(numpy.searchsorted(sentence_boundaries_in_tokens, tok_pos, side="left"))
            return min(idx + 1, len(sentences))

        # depth_scores are defined over gaps (len ~= len(tokseqs)-1). We need i and i+lag-1 to be valid gaps.
        max_i = min(len(lag_scores) - 1, len(depth_scores) - lag)
        for i in range(0, max_i + 1):
            j = i + lag
            left_gap = i
            right_gap = i + lag - 1

            if lag_scores[i] < sim_thresh:
                continue
            if left_gap >= len(depth_scores) or right_gap >= len(depth_scores):
                continue
            if depth_scores[left_gap] < dep_thresh or depth_scores[right_gap] < dep_thresh:
                continue

            # Determine the same window_size used by `_block_comparison_with_lag` at this i.
            n = len(tokseqs)
            left_cap = i + 1
            right_cap = n - (i + lag)
            window_size = min(self.k, left_cap, right_cap)
            if window_size <= 0:
                continue

            # Left/right blocks in tokseq index space
            left_start_tok = i - window_size + 1
            left_end_tok = i
            right_start_tok = i + lag
            right_end_tok = i + lag + window_size - 1

            # Convert tokseq indices to token positions (in tokens, not chars)
            left_start_tokpos = left_start_tok * self.w
            left_end_tokpos = (left_end_tok + 1) * self.w
            right_start_tokpos = right_start_tok * self.w
            right_end_tokpos = (right_end_tok + 1) * self.w

            # Map token positions to sentence boundary indices
            left_start_sentence = tokpos_to_sent_boundary(left_start_tokpos)
            left_end_sentence = tokpos_to_sent_boundary(left_end_tokpos)
            right_start_sentence = tokpos_to_sent_boundary(right_start_tokpos)
            right_end_sentence = tokpos_to_sent_boundary(right_end_tokpos)

            # Ensure proper ordering and non-empty blocks (in sentence space)
            if not (left_start_sentence < left_end_sentence <= right_start_sentence < right_end_sentence):
                continue

            # Enforce an explicit GAP between the blocks in sentence space.
            # GAP sentences are `sentences[left_end_sentence:right_start_sentence]`.
            if (right_start_sentence - left_end_sentence) < min_gap_sentences:
                continue

            results.append((left_start_sentence, left_end_sentence, right_start_sentence, right_end_sentence))

        return results

    def _block_comparison(self, tokseqs: list['TokenSequence'], token_table: dict[str, 'TokenTableField']) -> list[float]:
        def blk_frq(tok, block_set):
            count = 0
            for seq_id, freq in token_table[tok].ts_occurences:
                if seq_id in block_set:
                    count += freq
            return count
        
        gap_scores = []
        numgaps = len(tokseqs) - 1
        
        for curr_gap in range(numgaps):
            score_dividend, score_divisor_b1, score_divisor_b2 = 0.0, 0.0, 0.0
            
            if curr_gap < self.k - 1:
                window_size = curr_gap + 1
            elif curr_gap > numgaps - self.k:
                window_size = numgaps - curr_gap
            else:
                window_size = self.k
            
            b1_set = {ts.index for ts in tokseqs[curr_gap - window_size + 1:curr_gap + 1]}
            b2_set = {ts.index for ts in tokseqs[curr_gap + 1:curr_gap + window_size + 1]}
            
            for t in token_table:
                freq1 = blk_frq(t, b1_set)
                freq2 = blk_frq(t, b2_set)
                
                score_dividend += freq1 * freq2
                score_divisor_b1 += freq1 ** 2
                score_divisor_b2 += freq2 ** 2
            
            if score_divisor_b1 > 0 and score_divisor_b2 > 0:
                score = score_dividend / math.sqrt(score_divisor_b1 * score_divisor_b2)
            else:
                score = 0.0
            gap_scores.append(score)
            
        return gap_scores

    def _smooth_scores(self, gap_scores: list[float]) -> list[float]:
        return list(smooth(numpy.array(gap_scores[:]), window_len=self.smoothing_width + 1))

    def _depth_scores(self, scores: list[float]) -> list[float]:
        depth_scores = [0] * len(scores)
        clip = min(max(len(scores) // 10, 2), 5)
        
        for i in range(clip, len(scores) - clip):
            lpeak = scores[i]
            for score in scores[i::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            
            rpeak = scores[i]
            for score in scores[i:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
                
            depth_scores[i] = lpeak + rpeak - 2 * scores[i]
        return depth_scores

    def _identify_boundaries(self, depth_scores: list[float]) -> list[int]:
        boundaries = [0] * len(depth_scores)
        if not depth_scores:
            return boundaries
        
        avg = numpy.mean(depth_scores)
        stdev = numpy.std(depth_scores)
        
        if self.cutoff_policy == LC:
            cutoff = avg - stdev
        else:  # HC
            cutoff = avg - stdev / 2.0
        
        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))), reverse=True)
        hp = [x for x in depth_tuples if x[0] > cutoff]
        
        for dt in hp:
            boundaries[dt[1]] = 1
            for dt2 in hp:
                if dt[1] != dt2[1] and abs(dt2[1] - dt[1]) < 4 and boundaries[dt2[1]] == 1:
                    boundaries[dt[1]] = 0
                    break
        return boundaries


class TokenTableField:
    def __init__(self, first_pos: int, ts_occurences: int, total_count: int, last_tok_seq: int):
        self.first_pos = first_pos
        self.ts_occurences = ts_occurences
        self.total_count = total_count
        self.last_tok_seq = last_tok_seq


class TokenSequence:
    def __init__(self, index: int, wrdindex_list: list[int]):
        self.index = index
        self.wrdindex_list = wrdindex_list


def smooth(x: numpy.ndarray, window_len: int = 11) -> numpy.ndarray:
    if x.ndim != 1 or x.size < window_len or window_len < 3:
        return x
    s = numpy.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = numpy.ones(window_len, "d")
    y = numpy.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1:-window_len + 1]


if __name__ == "__main__":
    # example usage
    tt = TextTilingTokenizer(w=20, k=10)
    with open("data/travel_guides/tokyo.txt", "r") as f:
        sample_text = f.read()

    nest_spans = tt.find_nesting(sample_text, min_gap_sentences=1, lag=4)
    print("Nested spans:", nest_spans)

    sentences = tt._sentence_tokenize(sample_text)
    print("\nIndexed sentences (document):")
    for idx, sent in enumerate(sentences):
        # Print a compact one-line preview per sentence
        preview = " ".join(sent.split())
        print(f"[{idx}] {preview}")
    for (ls, le, rs, re_) in nest_spans:
        print("\n--- span ---")
        print("left_start,left_end,right_start,right_end =", (ls, le, rs, re_))
        print("LEFT sentences count =", le - ls)
        print("RIGHT sentences count =", re_ - rs)
        print("GAP sentences count =", max(0, rs - le))

        left_text = " ".join(sentences[ls:le])
        gap_text = " ".join(sentences[le:rs])
        right_text = " ".join(sentences[rs:re_])

        print("LEFT:", left_text)
        if gap_text:
            print("GAP:", gap_text)
        else:
            print("GAP: <empty>")
        print("RIGHT:", right_text)

    # Sentence list (no segmentation)
    sentences = tt._sentence_tokenize(sample_text)

    # Write sentences to CSV
    output_csv = "sentences.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence_id", "text"])
        writer.writeheader()
        writer.writerows({"sentence_id": i, "text": s} for i, s in enumerate(sentences))

    print(f"Saved {len(sentences)} sentences to {output_csv}")

    import csv
    import re

    def clean_for_csv(s: str) -> str:
        # turn any whitespace runs (including newlines/tabs) into a single space
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    sentences = tt._sentence_tokenize(sample_text)

    with open("sentences_google_sheets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["sentence_id", "text"])
        for i, s in enumerate(sentences):
            writer.writerow([i, clean_for_csv(s)])
