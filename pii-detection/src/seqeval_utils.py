"""
Sequence Evaluation Utilities for Named Entity Recognition
Enhanced seqeval metrics with improved entity extraction and evaluation.
"""

import warnings
from collections import defaultdict
from typing import List, Optional, Tuple
import numpy as np
from seqeval.metrics.v1 import SCORES, _precision_recall_fscore_support


def precision_recall_fscore_support(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    *,
    average: Optional[str] = None,
    warn_for=('precision', 'recall', 'f-score'),
    beta: float = 1.0,
    sample_weight: Optional[List[int]] = None,
    zero_division: str = 'warn',
    suffix: bool = False
) -> SCORES:
    """
    Compute precision, recall, F-measure and support for each class.

    Args:
        y_true: 2d array. Ground truth (correct) target values.
        y_pred: 2d array. Estimated targets as returned by a tagger.
        average: string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned.
        beta: float, 1.0 by default. The strength of recall versus precision in the F-score.
        warn_for: tuple or set, for internal use
        sample_weight: array-like of shape (n_samples,), default=None
        zero_division: "warn", 0 or 1, default="warn"
        suffix: bool, False by default.

    Returns:
        Tuple of (precision, recall, fbeta_score, support)

    Example:
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
        (0.5, 0.5, 0.5, 2)
    """

    def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
        """Extract true positives, actual and correct counts for evaluation."""
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        
        for type_name, start, end in get_entities(y_true, suffix):
            entities_true[type_name].add((start, end))
        for type_name, start, end in get_entities(y_pred, suffix):
            entities_pred[type_name].add((start, end))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


def get_entities(seq: List[str], suffix: bool = False) -> List[Tuple[str, int, int]]:
    """
    Gets entities from sequence.

    Args:
        seq: sequence of labels.
        suffix: whether to use suffix format for NE tags.

    Returns:
        List of (chunk_type, chunk_start, chunk_end) tuples.

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    def _validate_chunk(chunk, suffix):
        """Validate NE tag format."""
        if chunk in ['O', 'B', 'I', 'E', 'S', 'L', 'U']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S', '-L', '-U')):
                warnings.warn(f'{chunk} seems not to be NE tag.')
        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-', 'L-', 'U-')):
                warnings.warn(f'{chunk} seems not to be NE tag.')

    # Handle nested lists
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    """
    Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        bool: True if chunk ended.
    """
    chunk_end = False

    if prev_tag in ['E', 'L', 'S', 'U']:
        chunk_end = True

    if prev_tag == 'B' and tag in ['B', 'S', 'U', 'O']:
        chunk_end = True
    if prev_tag == 'I' and tag in ['B', 'S', 'U', 'O']:
        chunk_end = True

    if prev_tag not in ['O', '.'] and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    """
    Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        bool: True if chunk started.
    """
    chunk_start = False

    if tag in ['B', 'S', 'U']:
        chunk_start = True

    if prev_tag in ['E', 'L'] and tag in ['E', 'I', 'L']:
        chunk_start = True
    if prev_tag in ['S', 'U'] and tag in ['E', 'I', 'L']:
        chunk_start = True
    if prev_tag == 'O' and tag in ['E', 'I', 'L']:
        chunk_start = True

    if tag not in ['O', '.'] and prev_type != type_:
        chunk_start = True

    return chunk_start


def apply_seqeval_patches():
    """
    Apply monkey patches to seqeval for improved functionality.
    Call this function after importing seqeval to use enhanced metrics.
    """
    import seqeval.metrics.sequence_labeling
    
    seqeval.metrics.sequence_labeling.precision_recall_fscore_support = precision_recall_fscore_support
    seqeval.metrics.sequence_labeling.get_entities = get_entities
    seqeval.metrics.sequence_labeling.end_of_chunk = end_of_chunk
    seqeval.metrics.sequence_labeling.start_of_chunk = start_of_chunk
    
    print("✅ Seqeval patches applied successfully!")
