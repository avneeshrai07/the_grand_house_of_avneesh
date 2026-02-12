"""
The Grand House of Avneesh - Text Summarization Module
Exports main classes and utilities for text summarization
"""

from .text_summarizer import TextSummarizer
from .exceptions import SummarizationError, NLPProcessingError

__version__ = "1.0.0"
__all__ = ["TextSummarizer", "SummarizationError", "NLPProcessingError"]
