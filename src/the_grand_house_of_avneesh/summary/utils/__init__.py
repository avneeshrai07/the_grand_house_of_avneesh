"""
Text Summarization Utils Package
Modular NLP summarization utilities
"""

from .nlp.text_processor import parse_structure, extract_facts
from .nlp.sentence_ranker import rank_and_filter_sentences
from .llm.bedrock_client import BedrockClient
from .llm.synthesis import synthesize_summary
from .metrics.calculator import calculate_metrics
from .config.settings import SummarizerConfig

__all__ = [
    'parse_structure',
    'extract_facts',
    'rank_and_filter_sentences',
    'BedrockClient',
    'synthesize_summary',
    'calculate_metrics',
    'SummarizerConfig'
]

__version__ = '1.0.0'
