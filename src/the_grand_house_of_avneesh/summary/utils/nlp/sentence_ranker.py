"""
Sentence ranking and filtering using TF-IDF scores
Implements adaptive threshold-based selection
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


async def rank_and_filter_sentences(
    sentences: List[str],
    score_threshold: float = 0.7,
    min_sentences: int = 3,
    max_sentences: int = 15,
    max_features: int = 100,
    ngram_range: Tuple[int, int] = (1, 2),
    debug: bool = False
) -> List[str]:
    """
    Stage 3: Rank sentences by TF-IDF scores and filter adaptively

    Args:
        sentences: List of sentences to rank
        score_threshold: Normalized threshold for sentence selection (0.0-1.0)
        min_sentences: Minimum number of sentences to include
        max_sentences: Maximum number of sentences to include
        max_features: Maximum features for TF-IDF vectorizer
        ngram_range: N-gram range for TF-IDF (default: unigrams and bigrams)
        debug: Enable debug logging

    Returns:
        List of selected sentences in original order
    """
    try:
        logger.info("="*80)
        logger.info("📊 STAGE 3: ADAPTIVE RANK & FILTER")
        logger.info("="*80)
        logger.info(f"✓ Input sentences: {len(sentences)}")
        logger.info(f"✓ Selection strategy: Adaptive threshold-based")
        logger.info(f"✓ Range: {min_sentences}-{max_sentences} sentences")

        # Edge case: few sentences
        if len(sentences) <= min_sentences:
            logger.info("  → All sentences included (below minimum)")
            return sentences

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range
        )

        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(sentences)
        logger.info(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Calculate scores (sum of TF-IDF values per sentence)
        sentence_scores = tfidf_matrix.sum(axis=1).A1

        # Normalize scores to 0-1 range
        if sentence_scores.max() > 0:
            normalized_scores = (sentence_scores - sentence_scores.min()) / \
                              (sentence_scores.max() - sentence_scores.min())
        else:
            normalized_scores = sentence_scores

        logger.info(f"✓ Sentence scores computed and normalized")

        if debug:
            logger.info(f"  → Raw score statistics:")
            logger.info(f"     Min: {sentence_scores.min():.4f}")
            logger.info(f"     Max: {sentence_scores.max():.4f}")
            logger.info(f"     Mean: {sentence_scores.mean():.4f}")
            logger.info(f"     Median: {np.median(sentence_scores):.4f}")
            logger.info(f"     Std: {sentence_scores.std():.4f}")

        # Adaptive threshold: select sentences above threshold
        threshold_value = score_threshold
        selected_indices = np.where(normalized_scores >= threshold_value)[0]

        # Enforce min/max constraints
        if len(selected_indices) < min_sentences:
            # Select top N if below minimum
            top_indices = sentence_scores.argsort()[-min_sentences:][::-1]
            selected_indices = sorted(top_indices)
            logger.info(f"  ⚠️ Below minimum, selected top {min_sentences} sentences")
        elif len(selected_indices) > max_sentences:
            # Take only top N if above maximum
            top_from_selected = sentence_scores[selected_indices].argsort()[-max_sentences:][::-1]
            selected_indices = sorted(selected_indices[top_from_selected])
            logger.info(f"  ⚠️ Above maximum, limited to top {max_sentences} sentences")
        else:
            selected_indices = sorted(selected_indices)
            logger.info(f"  ✓ Selected {len(selected_indices)} sentences above threshold {threshold_value}")

        # Get selected sentences in original order
        top_sentences = [sentences[i] for i in selected_indices]

        if debug:
            logger.info(f"  → Selected {len(selected_indices)} sentences:")
            for i, idx in enumerate(selected_indices[:10]):  # Show first 10
                score = sentence_scores[idx]
                norm_score = normalized_scores[idx]
                logger.info(f"     [{i+1}] Score: {score:.4f} (norm: {norm_score:.2f}) | "
                          f"{sentences[idx][:70]}...")

        return top_sentences

    except Exception as e:
        logger.error(f"❌ Ranking failed: {str(e)}")
        raise


def calculate_tfidf_scores(
    sentences: List[str],
    max_features: int = 100,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate TF-IDF scores for sentences

    Args:
        sentences: List of sentences
        max_features: Maximum features for vectorizer
        ngram_range: N-gram range

    Returns:
        Tuple of (raw_scores, normalized_scores)
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=ngram_range
    )

    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Normalize
    if sentence_scores.max() > 0:
        normalized_scores = (sentence_scores - sentence_scores.min()) / \
                          (sentence_scores.max() - sentence_scores.min())
    else:
        normalized_scores = sentence_scores

    return sentence_scores, normalized_scores


def select_top_sentences(
    sentences: List[str],
    scores: np.ndarray,
    n: int
) -> List[str]:
    """
    Select top N sentences by score, maintaining original order

    Args:
        sentences: List of sentences
        scores: Array of scores
        n: Number of sentences to select

    Returns:
        List of top N sentences in original order
    """
    n = min(n, len(sentences))
    top_indices = scores.argsort()[-n:][::-1]
    top_indices = sorted(top_indices)
    return [sentences[i] for i in top_indices]
