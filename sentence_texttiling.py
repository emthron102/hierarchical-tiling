import math
import re
from nltk.corpus import stopwords as sw
import numpy
import spacy

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
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # Count only alphabetic tokens (excluding numbers) to match nopunct_text processing
        sent_token_counts = []
        for sent in sentences:
            tokens = re.findall(r"[a-zA-Z]+", sent)
            sent_token_counts.append(len(tokens))
        
        sentence_boundaries_in_tokens = numpy.cumsum(sent_token_counts)
        
        full_text_for_processing = "\n\n".join(sentences)
        
        lowercase_text = full_text_for_processing.lower()
        nopunct_text = "".join(c for c in lowercase_text if re.match(r"[a-z\-' \n\t]", c))
        
        tokseqs = self._divide_to_tokensequences(nopunct_text)
        
        if len(tokseqs) < 2:
            return [text]

        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list if wi[0] not in self.stopwords]

        token_table = self._create_token_table(tokseqs)
        
        gap_scores = self._block_comparison(tokseqs, token_table)
        
        if len(gap_scores) < 3:
            return [text]
            
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
        
        segments = []
        for start, end in zip(final_sentence_indices[:-1], final_sentence_indices[1:]):
            segment_sentences = sentences[start:end]
            segments.append(" ".join(segment_sentences))
            
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

    segments = tt.tokenize(sample_text)

    print(f"Total segments: {len(segments)}")
    for i, seg in enumerate(segments, 1):
        print(f"\n[Segment {i}]")
        print(seg)
