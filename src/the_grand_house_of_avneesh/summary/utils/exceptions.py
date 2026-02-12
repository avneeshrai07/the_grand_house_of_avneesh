"""
Custom exceptions for text summarization pipeline
"""


class SummarizationError(Exception):
    """Base exception for summarization errors"""
    pass


class NLPProcessingError(SummarizationError):
    """Exception raised for NLP processing errors"""
    pass


class LLMError(SummarizationError):
    """Exception raised for LLM-related errors"""
    pass


class ConfigurationError(SummarizationError):
    """Exception raised for configuration errors"""
    pass


class ValidationError(SummarizationError):
    """Exception raised for validation errors"""
    pass
