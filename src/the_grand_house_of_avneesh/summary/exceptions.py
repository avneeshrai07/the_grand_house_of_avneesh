"""
Custom Exceptions for Text Summarization Module
Provides specific error types for different failure modes
"""


class SummarizationError(Exception):
    """
    Base exception for summarization-related errors
    
    Raised when:
    - AWS Bedrock API fails
    - Empty or invalid LLM responses
    - Pipeline execution errors
    """
    pass


class NLPProcessingError(Exception):
    """
    Exception for NLP processing failures
    
    Raised when:
    - SpaCy model loading fails
    - Text parsing errors
    - Entity extraction failures
    - TF-IDF vectorization errors
    """
    pass


class ConfigurationError(Exception):
    """
    Exception for configuration-related errors
    
    Raised when:
    - Missing environment variables
    - Invalid model configurations
    - AWS credentials issues
    """
    pass
