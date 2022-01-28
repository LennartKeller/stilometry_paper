from typing import Dict, Hashable, List, Tuple
import numpy as np
from more_itertools import chunked

class Zeta:


    def __init__(self, chunk_size: int = 50):
        self.chunk_size = chunk_size


    def _chunk_texts(self, texts, labels):
        """
        Chunks texts into evenly sized parts.
        Since we are only interest in occurences not in counts (?) each chunked is converted to a set
        Also we copy the labels accordingly.
        """
        total_chunks = []
        total_labels = []
        for text, label in zip(texts, labels):
            chunks = [set(c) for c in chunked(text, self.chunk_size)]
            new_labels = [label] * len(chunks)
            total_chunks.extend(chunks)
            total_labels.extend(new_labels)
        return np.array(total_chunks), np.array(total_labels)


    @staticmethod
    def _get_types(texts: List[List[str]]) -> Tuple[Dict[str, int], np.array]:
        types2idx = {}
        vocab = []
        for idx, ty in enumerate(sorted(set([token for text in texts for token in text]))):
            types2idx[ty] = idx
            vocab.append(ty)
        return types2idx, np.array(vocab, dtype='object')


    def _compute_occ_scores(self, chunks, labels):

        data = {label: np.zeros(len(self.vocab_), dtype='float64') for label in set(labels)}

        for chunk, label in zip(chunks, labels):
            for ty in chunk:
                data[label][self.type2idx_[ty]] += 1.0

        for label in data:
            n_partitions = len(chunks[labels == label])
            data[label] /= n_partitions
        return data



    def fit(self, X: List[List[str]], y: List[Hashable], target_partition: Hashable = None):
        """
        Inputs:
        X: Tokenized Texts as List of list of strings
        y: Labels for each text showing the partition of the texts
        """
        if len(set(y)) != 2:
            raise Exception("Number of unique labels has to be 2")
        if len(X) != len(y):
            raise Exception('X and y do not have the same length..')

        if target_partition is None:
            target_partition = y[0]

        # 1. Get list of types over all texts
        self.type2idx_, self.vocab_ = self._get_types(X)
        # 2. Chunk each text into parts of size self.window_size
        # Copy labels accordingly
        chunks, labels = self._chunk_texts(X, y)
        # 3. For each type count number of chunks where it appears and divide by total number of chunks for each label.
        # Subtract scores of target partition from scores of other parition => Return Scores
        occ_scores = self._compute_occ_scores(chunks, labels)
        target = occ_scores[target_partition]

        other_label = list(occ_scores.keys())
        other_label.remove(target_partition)
        other_label = other_label[0]
        other = occ_scores[other_label]

        return target - other