"""
Metrics calculation utilities for summarization performance
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(original: str, summary: str) -> Dict[str, Any]:
    """
    Calculate comprehensive compression and quality metrics

    Args:
        original: Original input text
        summary: Generated summary text

    Returns:
        Dictionary containing various metrics
    """
    logger.info("="*80)
    logger.info("📐 CALCULATING METRICS")
    logger.info("="*80)

    # Character counts
    original_chars = len(original)
    compressed_chars = len(summary)

    # Word counts
    original_words = len(original.split())
    compressed_words = len(summary.split())

    # Sentence counts
    original_sentences = len([s for s in original.split('.') if s.strip()])
    compressed_sentences = len([s for s in summary.split('.') if s.strip()])

    # Calculate reduction
    char_reduction = ((original_chars - compressed_chars) / original_chars) * 100 if original_chars > 0 else 0
    word_reduction = ((original_words - compressed_words) / original_words) * 100 if original_words > 0 else 0

    # Calculate compression ratio
    compression_ratio = compressed_chars / original_chars if original_chars > 0 else 0

    metrics = {
        # Character metrics
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "char_reduction_percentage": f"{char_reduction:.1f}%",

        # Word metrics
        "original_words": original_words,
        "compressed_words": compressed_words,
        "word_reduction_percentage": f"{word_reduction:.1f}%",

        # Sentence metrics
        "original_sentences": original_sentences,
        "compressed_sentences": compressed_sentences,

        # Ratios
        "compression_ratio": f"{compression_ratio:.2f}x",
        "compression_ratio_value": compression_ratio,

        # Summary stats
        "avg_chars_per_sentence_original": original_chars / original_sentences if original_sentences > 0 else 0,
        "avg_chars_per_sentence_summary": compressed_chars / compressed_sentences if compressed_sentences > 0 else 0,
        "avg_words_per_sentence_original": original_words / original_sentences if original_sentences > 0 else 0,
        "avg_words_per_sentence_summary": compressed_words / compressed_sentences if compressed_sentences > 0 else 0,
    }

    logger.info(f"✓ Metrics calculated:")
    logger.info(f"  → Original: {original_chars:,} chars, {original_words} words, {original_sentences} sentences")
    logger.info(f"  → Summary: {compressed_chars:,} chars, {compressed_words} words, {compressed_sentences} sentences")
    logger.info(f"  → Character reduction: {char_reduction:.1f}%")
    logger.info(f"  → Word reduction: {word_reduction:.1f}%")
    logger.info(f"  → Compression ratio: {compression_ratio:.2f}x")

    return metrics


def calculate_simple_metrics(original: str, summary: str) -> Dict[str, Any]:
    """
    Calculate basic metrics (chars, words, reduction)

    Args:
        original: Original text
        summary: Summary text

    Returns:
        Dictionary with basic metrics
    """
    original_chars = len(original)
    original_words = len(original.split())
    compressed_chars = len(summary)
    compressed_words = len(summary.split())

    reduction_percentage = ((original_chars - compressed_chars) / original_chars) * 100 if original_chars > 0 else 0
    compression_ratio = compressed_chars / original_chars if original_chars > 0 else 0

    return {
        "original_chars": original_chars,
        "original_words": original_words,
        "compressed_chars": compressed_chars,
        "compressed_words": compressed_words,
        "reduction_percentage": f"{reduction_percentage:.1f}%",
        "compression_ratio": f"{compression_ratio:.2f}x"
    }


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format metrics into readable report string

    Args:
        metrics: Metrics dictionary

    Returns:
        Formatted report string
    """
    report = []
    report.append("="*60)
    report.append("SUMMARIZATION METRICS REPORT")
    report.append("="*60)
    report.append(f"Original Text:")
    report.append(f"  - Characters: {metrics.get('original_chars', 'N/A'):,}")
    report.append(f"  - Words: {metrics.get('original_words', 'N/A')}")
    report.append(f"  - Sentences: {metrics.get('original_sentences', 'N/A')}")
    report.append(f"\nSummary Text:")
    report.append(f"  - Characters: {metrics.get('compressed_chars', 'N/A'):,}")
    report.append(f"  - Words: {metrics.get('compressed_words', 'N/A')}")
    report.append(f"  - Sentences: {metrics.get('compressed_sentences', 'N/A')}")
    report.append(f"\nCompression:")
    report.append(f"  - Character reduction: {metrics.get('char_reduction_percentage', 'N/A')}")
    report.append(f"  - Word reduction: {metrics.get('word_reduction_percentage', 'N/A')}")
    report.append(f"  - Compression ratio: {metrics.get('compression_ratio', 'N/A')}")
    report.append("="*60)

    return "\n".join(report)


def validate_summary_quality(
    metrics: Dict[str, Any],
    min_compression: float = 0.3,
    max_compression: float = 0.8
) -> Tuple[bool, List[str]]:
    """
    Validate summary quality based on metrics

    Args:
        metrics: Metrics dictionary
        min_compression: Minimum acceptable compression ratio
        max_compression: Maximum acceptable compression ratio

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    compression_ratio = metrics.get('compression_ratio_value', 0)

    if compression_ratio < min_compression:
        issues.append(f"Summary too short (compression: {compression_ratio:.2f}, min: {min_compression})")

    if compression_ratio > max_compression:
        issues.append(f"Summary too long (compression: {compression_ratio:.2f}, max: {max_compression})")

    if metrics.get('compressed_words', 0) < 50:
        issues.append("Summary has fewer than 50 words")

    return len(issues) == 0, issues
