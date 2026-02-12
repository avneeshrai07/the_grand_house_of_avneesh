"""
Main TextSummarizer class that orchestrates the 4-stage pipeline
Uses modular utility functions from utils package
"""

import spacy
from typing import Dict, Any
import logging
from datetime import datetime

from utils.config.settings import SummarizerConfig
from utils.nlp.text_processor import parse_structure, extract_facts
from utils.nlp.sentence_ranker import rank_and_filter_sentences
from utils.llm.bedrock_client import BedrockClient
from utils.llm.synthesis import synthesize_summary
from utils.metrics.calculator import calculate_metrics
from utils.exceptions import SummarizationError, NLPProcessingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextSummarizer:
    """
    Text summarization pipeline with 4 stages + dynamic sentence selection:
    1. Parse Structure (rule-based)
    2. Extract Facts (NLP/SpaCy)
    3. Rank & Filter (TF-IDF with dynamic threshold)
    4. LLM Synthesis (ONE call with enhanced prompt)
    """

    def __init__(self, config: SummarizerConfig = None):
        """
        Initialize the summarizer with configuration

        Args:
            config: SummarizerConfig instance (uses defaults if None)
        """
        self.config = config or SummarizerConfig()

        logger.info("=" * 80)
        logger.info("🚀 INITIALIZING ADAPTIVE TEXT SUMMARIZER")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config.model_id}")
        logger.info(f"Region: {self.config.region_name}")
        logger.info(f"Score Threshold: {self.config.score_threshold} (adaptive)")
        logger.info(f"Sentence Range: {self.config.min_sentences}-{self.config.max_sentences}")
        logger.info(f"Temperature: {self.config.temperature}")
        logger.info(f"Max Tokens: {self.config.max_tokens}")
        logger.info(f"Debug Mode: {self.config.debug}")

        # Initialize SpaCy
        try:
            logger.info(f"📚 Loading SpaCy model: {self.config.spacy_model}...")
            self.nlp = spacy.load(self.config.spacy_model)
            logger.info("✅ SpaCy model loaded successfully")
        except OSError:
            raise NLPProcessingError(
                f"SpaCy model '{self.config.spacy_model}' not found. "
                f"Install it with: python -m spacy download {self.config.spacy_model}"
            )

        # Initialize AWS Bedrock client
        try:
            logger.info("☁️ Initializing AWS Bedrock client...")
            self.bedrock_client = BedrockClient(
                model_id=self.config.model_id,
                region_name=self.config.region_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            logger.info(f"✅ Bedrock client initialized: {self.config.model_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bedrock: {str(e)}")
            raise SummarizationError(f"Failed to initialize AWS Bedrock client: {str(e)}")

        logger.info("=" * 80)

    async def summarize(self, text: str) -> Dict[str, Any]:
        """
        Main summarization pipeline with step-by-step processing

        Args:
            text: Input text to summarize

        Returns:
            Dictionary containing summary, metrics, and debug info
        """
        logger.info("\n" + "🔥" * 40)
        logger.info("STARTING ADAPTIVE SUMMARIZATION PIPELINE")
        logger.info("🔥" * 40)
        logger.info(f"Input Text Length: {len(text)} characters")
        logger.info(f"Input Text Preview: {text[:200]}...")

        try:
            # Stage 1: Parse Structure
            paragraphs, sentences = await parse_structure(
                text=text,
                nlp_model=self.nlp,
                debug=self.config.debug
            )

            # Stage 2: Extract Facts
            entities, key_phrases = await extract_facts(
                text=text,
                nlp_model=self.nlp,
                debug=self.config.debug
            )

            # Stage 3: Adaptive Rank & Filter
            top_sentences = await rank_and_filter_sentences(
                sentences=sentences,
                score_threshold=self.config.score_threshold,
                min_sentences=self.config.min_sentences,
                max_sentences=self.config.max_sentences,
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                debug=self.config.debug
            )

            # Stage 4: Enhanced LLM Synthesis
            summary, token_usage = await synthesize_summary(
                bedrock_client=self.bedrock_client,
                sentences=top_sentences,
                entities=entities,
                key_phrases=key_phrases,
                original_text=text,
                min_summary_words=self.config.min_summary_words,
                max_summary_words=self.config.max_summary_words,
                summary_target_ratio=self.config.summary_target_ratio,
                debug=self.config.debug
            )

            # Calculate metrics
            metrics = calculate_metrics(text, summary)

            # Final summary
            logger.info("\n" + "🎯" * 40)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("🎯" * 40)
            logger.info(f"Original: {metrics['original_chars']} chars")
            logger.info(f"Summary: {metrics['compressed_chars']} chars")
            logger.info(f"Reduction: {metrics['char_reduction_percentage']}")
            logger.info(f"Compression Ratio: {metrics['compression_ratio']}")
            logger.info("🎯" * 40 + "\n")

            return {
                "summary": summary,
                "metrics": metrics,
                "token_usage": token_usage,
                "debug_info": {
                    "total_paragraphs": len(paragraphs),
                    "total_sentences": len(sentences),
                    "selected_sentences": len(top_sentences),
                    "total_entities": len(entities),
                    "total_key_phrases": len(key_phrases),
                    "selection_method": "adaptive_threshold",
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise SummarizationError(f"Summarization pipeline failed: {str(e)}")


def create_summarizer(config: SummarizerConfig = None) -> TextSummarizer:
    """
    Factory function to create TextSummarizer instance

    Args:
        config: SummarizerConfig instance

    Returns:
        TextSummarizer instance
    """
    return TextSummarizer(config)
