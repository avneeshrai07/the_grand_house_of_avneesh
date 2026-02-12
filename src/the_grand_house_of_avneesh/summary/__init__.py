from .text_summarizer import TextSummarizer, create_summarizer
from .utils.exceptions import (
    SummarizationError,
    NLPProcessingError,
    LLMError,
    ValidationError,
    ConfigurationError
)
from .utils.config.settings import SummarizerConfig

__version__ = "1.0.0"
__author__ = "The Grand House of Avneesh"
__email__ = "contact@grandhouse.dev"

__all__ = [
    # Main classes and functions
    "TextSummarizer",
    "create_summarizer"
]
