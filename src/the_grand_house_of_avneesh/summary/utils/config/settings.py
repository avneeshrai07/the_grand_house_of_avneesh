"""
Configuration settings for the text summarizer
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SummarizerConfig:
    """Configuration class for TextSummarizer"""

    # Sentence selection parameters
    score_threshold: float = 0.7
    min_sentences: int = 3
    max_sentences: int = 15

    # AWS Bedrock settings
    model_id: str = os.getenv("MODEL_ID", "amazon.nova-lite-v1:0")
    region_name: str = os.getenv("AWS_REGION", "ap-southeast-2")

    # LLM parameters
    temperature: float = 0.3
    max_tokens: int = 3096

    # Feature flags
    debug: bool = True

    # NLP settings
    spacy_model: str = "en_core_web_sm"
    tfidf_max_features: int = 100
    tfidf_ngram_range: tuple = (1, 2)

    # Summary constraints
    min_summary_words: int = 150
    max_summary_words: int = 500
    summary_target_ratio: float = 0.25  # 25% of original

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")

        if self.min_sentences > self.max_sentences:
            raise ValueError("min_sentences cannot be greater than max_sentences")

        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")

    @classmethod
    def from_env(cls) -> 'SummarizerConfig':
        """Create configuration from environment variables"""
        return cls(
            model_id=os.getenv("MODEL_ID", cls.model_id),
            region_name=os.getenv("AWS_REGION", cls.region_name),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("MAX_TOKENS", cls.max_tokens)),
            debug=os.getenv("DEBUG", "true").lower() == "true"
        )


def get_default_config() -> SummarizerConfig:
    """Get default configuration"""
    return SummarizerConfig()
