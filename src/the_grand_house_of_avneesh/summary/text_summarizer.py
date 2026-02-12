"""
TextSummarizer Library - Production-Ready Text Summarization API
A modular 4-stage NLP pipeline with AWS Bedrock LLM integration

Usage:
    from text_summarizer import create_summarizer

    result = await create_summarizer(
        aws_access_key="your-key",
        aws_secret_key="your-secret",
        text="Your long text to summarize...",
        model_id="amazon.nova-lite-v1:0",
        region_name="ap-southeast-2"
    )

    print(result["summary"])
"""

import spacy
import os
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from the_grand_house_of_avneesh.summary.utils.config.settings import SummarizerConfig
from the_grand_house_of_avneesh.summary.utils.nlp.text_processor import parse_structure, extract_facts
from the_grand_house_of_avneesh.summary.utils.nlp.sentence_ranker import rank_and_filter_sentences
from the_grand_house_of_avneesh.summary.utils.llm.bedrock_client import BedrockClient
from the_grand_house_of_avneesh.summary.utils.llm.synthesis import synthesize_summary
from the_grand_house_of_avneesh.summary.utils.metrics.calculator import calculate_metrics
from the_grand_house_of_avneesh.summary.utils.exceptions import (
    SummarizationError, 
    NLPProcessingError,
    ValidationError,
    ConfigurationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextSummarizer:
    """
    Production-ready text summarization pipeline with 4 stages:
    1. Parse Structure (rule-based paragraph/sentence splitting)
    2. Extract Facts (NLP entity & key phrase extraction)
    3. Rank & Filter (TF-IDF adaptive sentence selection)
    4. LLM Synthesis (AWS Bedrock summarization)
    """

    def __init__(self, config: Optional[SummarizerConfig] = None):
        """
        Initialize the summarizer with configuration.

        Args:
            config: SummarizerConfig instance. If None, uses defaults.

        Raises:
            ConfigurationError: If configuration is invalid
            NLPProcessingError: If SpaCy model cannot be loaded
            SummarizationError: If AWS Bedrock client fails to initialize
        """
        try:
            self.config = config or SummarizerConfig()
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {str(e)}")

        if self.config.debug:
            logger.info("=" * 80)
            logger.info("🚀 INITIALIZING TEXT SUMMARIZER")
            logger.info("=" * 80)
            logger.info(f"Model: {self.config.model_id}")
            logger.info(f"Region: {self.config.region_name}")
            logger.info(f"Score Threshold: {self.config.score_threshold}")
            logger.info(f"Sentence Range: {self.config.min_sentences}-{self.config.max_sentences}")
            logger.info(f"Temperature: {self.config.temperature}")

        # Initialize SpaCy
        self._initialize_spacy()

        # Initialize AWS Bedrock client
        self._initialize_bedrock()

        if self.config.debug:
            logger.info("=" * 80)
            logger.info("✅ Initialization complete")
            logger.info("=" * 80)

    def _initialize_spacy(self):
        """Initialize SpaCy NLP model with automatic download fallback."""
        try:
            if self.config.debug:
                logger.info(f"📚 Loading SpaCy model: {self.config.spacy_model}...")

            self.nlp = spacy.load(self.config.spacy_model)

            if self.config.debug:
                logger.info("✅ SpaCy model loaded successfully")

        except OSError:
            # Model not found - attempt automatic download
            logger.warning(f"⚠️  SpaCy model '{self.config.spacy_model}' not found. Downloading automatically...")
            
            try:
                import subprocess
                import sys
                
                logger.info(f"📦 Downloading {self.config.spacy_model}... (this may take a minute)")
                
                # Download the model
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", self.config.spacy_model],
                    stdout=subprocess.DEVNULL if not self.config.debug else None,
                    stderr=subprocess.DEVNULL if not self.config.debug else None
                )
                
                logger.info(f"✅ Successfully downloaded '{self.config.spacy_model}'")
                
                # Try loading again
                self.nlp = spacy.load(self.config.spacy_model)
                
                if self.config.debug:
                    logger.info("✅ SpaCy model loaded successfully after download")
                    
            except subprocess.CalledProcessError as e:
                error_msg = (
                    f"❌ Failed to automatically download SpaCy model '{self.config.spacy_model}'.\n"
                    f"\nPlease install it manually using:\n"
                    f"  python -m spacy download {self.config.spacy_model}\n"
                )
                logger.error(error_msg)
                raise NLPProcessingError(error_msg) from e
            except Exception as e:
                error_msg = f"❌ Unexpected error during SpaCy initialization: {str(e)}"
                logger.error(error_msg)
                raise NLPProcessingError(error_msg) from e


    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client with error handling."""
        try:
            if self.config.debug:
                logger.info("☁️  Initializing AWS Bedrock client...")

            self.bedrock_client = BedrockClient(
                model_id=self.config.model_id,
                region_name=self.config.region_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            if self.config.debug:
                logger.info(f"✅ Bedrock client initialized: {self.config.model_id}")

        except Exception as e:
            error_msg = (
                f"❌ Failed to initialize AWS Bedrock client.\n"
                f"\nError details: {str(e)}\n"
            )
            logger.error(error_msg)
            raise SummarizationError(error_msg) from e

    async def summarize(
        self, 
        text: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Summarize the input text using the 4-stage pipeline.

        Args:
            text: Input text to summarize. Must be non-empty string.
            custom_config: Optional dictionary to override config for this call only.

        Returns:
            Dictionary containing summary, metrics, token_usage, and debug_info

        Raises:
            ValidationError: If input text is invalid
            SummarizationError: If pipeline fails
        """
        # Validate input
        self._validate_input(text)

        # Apply custom config if provided
        if custom_config:
            config = self._merge_config(custom_config)
        else:
            config = self.config

        if config.debug:
            logger.info("\n" + "🔥" * 40)
            logger.info("STARTING SUMMARIZATION PIPELINE")
            logger.info("🔥" * 40)
            logger.info(f"Input Length: {len(text)} characters, {len(text.split())} words")

        try:
            # Stage 1: Parse Structure
            paragraphs, sentences = await parse_structure(
                text=text,
                nlp_model=self.nlp,
                debug=config.debug
            )

            # Stage 2: Extract Facts
            entities, key_phrases = await extract_facts(
                text=text,
                nlp_model=self.nlp,
                debug=config.debug
            )

            # Stage 3: Adaptive Rank & Filter
            top_sentences = await rank_and_filter_sentences(
                sentences=sentences,
                score_threshold=config.score_threshold,
                min_sentences=config.min_sentences,
                max_sentences=config.max_sentences,
                max_features=config.tfidf_max_features,
                ngram_range=config.tfidf_ngram_range,
                debug=config.debug
            )

            # Stage 4: Enhanced LLM Synthesis
            summary, token_usage = await synthesize_summary(
                bedrock_client=self.bedrock_client,
                sentences=top_sentences,
                entities=entities,
                key_phrases=key_phrases,
                original_text=text,
                min_summary_words=config.min_summary_words,
                max_summary_words=config.max_summary_words,
                summary_target_ratio=config.summary_target_ratio,
                debug=config.debug
            )

            # Calculate metrics
            metrics = calculate_metrics(text, summary)

            if config.debug:
                logger.info("\n" + "🎯" * 40)
                logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
                logger.info("🎯" * 40)
                logger.info(f"Compression: {metrics['compression_ratio']}")

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
            error_msg = f"Summarization pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise SummarizationError(error_msg) from e

    def _validate_input(self, text: str):
        """Validate input text."""
        if not isinstance(text, str):
            raise ValidationError(
                f"❌ Input must be a string, got {type(text).__name__}.\n"
                f"Please provide text as a string."
            )

        if not text or not text.strip():
            raise ValidationError(
                "❌ Input text is empty.\n"
                "Please provide non-empty text to summarize."
            )

        if len(text.strip()) < 50:
            raise ValidationError(
                f"❌ Input text too short ({len(text.strip())} characters).\n"
                f"Please provide at least 50 characters for meaningful summarization."
            )

        word_count = len(text.split())
        if word_count < 20:
            raise ValidationError(
                f"❌ Input text too short ({word_count} words).\n"
                f"Please provide at least 20 words for summarization."
            )

    def _merge_config(self, custom_config: Dict[str, Any]) -> SummarizerConfig:
        """Merge custom config with existing config."""
        config_dict = {
            'score_threshold': custom_config.get('score_threshold', self.config.score_threshold),
            'min_sentences': custom_config.get('min_sentences', self.config.min_sentences),
            'max_sentences': custom_config.get('max_sentences', self.config.max_sentences),
            'temperature': custom_config.get('temperature', self.config.temperature),
            'model_id': self.config.model_id,
            'region_name': self.config.region_name,
            'max_tokens': self.config.max_tokens,
            'debug': custom_config.get('debug', self.config.debug),
            'spacy_model': self.config.spacy_model,
            'tfidf_max_features': self.config.tfidf_max_features,
            'tfidf_ngram_range': self.config.tfidf_ngram_range,
            'min_summary_words': self.config.min_summary_words,
            'max_summary_words': self.config.max_summary_words,
            'summary_target_ratio': self.config.summary_target_ratio
        }
        return SummarizerConfig(**config_dict)


async def create_summarizer(
    text: str,
    aws_access_key: str,
    aws_secret_key: str,
    model_id: str = "amazon.nova-lite-v1:0",
    region_name: str = "ap-southeast-2",
    min_sentences: int = 3,
    max_sentences: int = 15,
    temperature: float = 0.3,
    score_threshold: float = 0.7,
    max_tokens: int = 3096,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Convenient factory function that summarizes text in one call.
    This function validates inputs, sets up AWS credentials, and returns the summary.

    Args:
        text (str, REQUIRED): Input text to summarize
            - Must be non-empty string
            - Minimum 50 characters
            - Minimum 20 words

        aws_access_key (str, optional): AWS Access Key ID
            - If not provided, uses AWS_ACCESS_KEY_ID environment variable

        aws_secret_key (str, optional): AWS Secret Access Key
            - If not provided, uses AWS_SECRET_ACCESS_KEY environment variable

        model_id (str, optional): AWS Bedrock model ID
            - Default: "amazon.nova-lite-v1:0"
            - Options: amazon.nova-lite-v1:0, amazon.nova-pro-v1:0, etc.

        region_name (str, optional): AWS region
            - Default: "ap-southeast-2"
            - Must be a region where Bedrock is available

        min_sentences (int, optional): Minimum sentences to include in summary
            - Default: 3
            - Must be >= 1

        max_sentences (int, optional): Maximum sentences to include in summary
            - Default: 15
            - Must be >= min_sentences

        temperature (float, optional): LLM creativity level
            - Default: 0.3
            - Range: 0.0 (focused) to 1.0 (creative)

        score_threshold (float, optional): TF-IDF threshold for sentence selection
            - Default: 0.7
            - Range: 0.0 to 1.0

        max_tokens (int, optional): Maximum tokens for LLM response
            - Default: 3096

        debug (bool, optional): Enable detailed logging
            - Default: False

    Returns:
        Dictionary containing:
            - summary (str): Generated summary text
            - metrics (dict): Compression metrics
            - token_usage (dict): Token usage statistics
            - debug_info (dict): Pipeline debugging information

    Raises:
        ValidationError: If required inputs are missing or invalid
        ConfigurationError: If AWS credentials are invalid
        NLPProcessingError: If SpaCy model is not installed
        SummarizationError: If summarization fails

    Example:
        >>> result = await create_summarizer(
        ...     text="Your long article text here...",
        ...     aws_access_key="AKIAIOSFODNN7EXAMPLE",
        ...     aws_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ...     temperature=0.5,
        ...     debug=True
        ... )
        >>> print(result["summary"])
        >>> print(f"Reduced to {result['metrics']['compression_ratio']}")
    """

    # Validate required parameter: text
    if not text:
        raise ValidationError(
            "❌ REQUIRED PARAMETER MISSING: 'text'\n"
            "\nThe 'text' parameter is required and cannot be empty.\n"
            "\nExample usage:\n"
            "  result = await create_summarizer(\n"
            "      text='Your long text to summarize...',\n"
            "      aws_access_key='your-key',\n"
            "      aws_secret_key='your-secret'\n"
            "  )\n"
        )

    # Validate and set AWS credentials
    if aws_access_key and aws_secret_key:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
    elif not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise ConfigurationError(
            "❌ AWS CREDENTIALS MISSING\n"
            "\nYou must provide AWS credentials either:\n"
            "\n1. As parameters:\n"
            "   result = await create_summarizer(\n"
            "       text='...',\n"
            "       aws_access_key='AKIAIOSFODNN7EXAMPLE',\n"
            "       aws_secret_key='wJalrXUtnFEMI/K7MDENG/...'\n"
            "   )\n"
            "\n2. As environment variables:\n"
            "   export AWS_ACCESS_KEY_ID='your-key'\n"
            "   export AWS_SECRET_ACCESS_KEY='your-secret'\n"
            "\n3. Using AWS CLI configuration:\n"
            "   aws configure\n"
        )

    # Validate parameter ranges
    if not 0.0 <= temperature <= 1.0:
        raise ValidationError(
            f"❌ INVALID PARAMETER: temperature = {temperature}\n"
            f"Temperature must be between 0.0 and 1.0\n"
            f"  • 0.0 = very focused and deterministic\n"
            f"  • 0.5 = balanced\n"
            f"  • 1.0 = very creative and random\n"
        )

    if not 0.0 <= score_threshold <= 1.0:
        raise ValidationError(
            f"❌ INVALID PARAMETER: score_threshold = {score_threshold}\n"
            f"Score threshold must be between 0.0 and 1.0\n"
        )

    if min_sentences < 1:
        raise ValidationError(
            f"❌ INVALID PARAMETER: min_sentences = {min_sentences}\n"
            f"Minimum sentences must be at least 1\n"
        )

    if max_sentences < min_sentences:
        raise ValidationError(
            f"❌ INVALID PARAMETERS:\n"
            f"  min_sentences = {min_sentences}\n"
            f"  max_sentences = {max_sentences}\n"
            f"\nmax_sentences must be >= min_sentences\n"
        )

    # Create configuration
    try:
        config = SummarizerConfig(
            score_threshold=score_threshold,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            model_id=model_id,
            region_name=region_name,
            temperature=temperature,
            max_tokens=max_tokens,
            debug=debug
        )
    except Exception as e:
        raise ConfigurationError(f"Failed to create configuration: {str(e)}")

    # Create summarizer and process
    try:
        summarizer = TextSummarizer(config)
        result = await summarizer.summarize(text)
        return result

    except ValidationError:
        raise
    except NLPProcessingError:
        raise
    except Exception as e:
        raise SummarizationError(
            f"❌ Summarization failed: {str(e)}\n"
            f"\nPlease check:\n"
            f"  1. AWS credentials are valid\n"
            f"  2. SpaCy model is installed: python -m spacy download en_core_web_sm\n"
            f"  3. Network connectivity\n"
            f"  4. Model '{model_id}' is available in region '{region_name}'\n"
        ) from e


