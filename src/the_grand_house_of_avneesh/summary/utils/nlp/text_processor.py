"""
Text processing utilities for structure parsing and fact extraction
Uses SpaCy for NLP operations
"""

import spacy
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text parsing and NLP operations"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize TextProcessor with SpaCy model

        Args:
            spacy_model: Name of the SpaCy model to load
        """
        try:
            logger.info(f"📚 Loading SpaCy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model)
            logger.info("✅ SpaCy model loaded successfully")
        except OSError:
            raise OSError(
                f"SpaCy model '{spacy_model}' not found. "
                f"Install it with: python -m spacy download {spacy_model}"
            )


async def parse_structure(text: str, nlp_model=None, debug: bool = False) -> Tuple[List[str], List[str]]:
    """
    Stage 1: Parse document structure into paragraphs and sentences

    Args:
        text: Input text to parse
        nlp_model: SpaCy model instance (if None, creates new one)
        debug: Enable debug logging

    Returns:
        Tuple of (paragraphs, sentences)
    """
    try:
        logger.info("="*80)
        logger.info("📝 STAGE 1: PARSE STRUCTURE")
        logger.info("="*80)

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        logger.info(f"✓ Paragraphs extracted: {len(paragraphs)}")

        if debug and paragraphs:
            logger.info(f"  → First paragraph: {paragraphs[0][:100]}...")
            if len(paragraphs) > 1:
                logger.info(f"  → Last paragraph: {paragraphs[-1][:100]}...")

        # Initialize SpaCy if not provided
        if nlp_model is None:
            nlp_model = spacy.load("en_core_web_sm")

        # Split into sentences using SpaCy
        doc = nlp_model(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        logger.info(f"✓ Sentences extracted: {len(sentences)}")

        if debug and sentences:
            logger.info(f"  → Sample sentences (first 3):")
            for i, sent in enumerate(sentences[:3]):
                logger.info(f"     [{i+1}] {sent[:70]}...")

        # Sentence length analysis
        avg_sent_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        logger.info(f"✓ Average sentence length: {avg_sent_length:.1f} chars")

        return paragraphs, sentences

    except Exception as e:
        logger.error(f"❌ Structure parsing failed: {str(e)}")
        raise


async def extract_facts(text: str, nlp_model=None, debug: bool = False) -> Tuple[List[Dict], List[str]]:
    """
    Stage 2: Extract named entities and key phrases from text

    Args:
        text: Input text to analyze
        nlp_model: SpaCy model instance (if None, creates new one)
        debug: Enable debug logging

    Returns:
        Tuple of (entities, key_phrases)
    """
    try:
        logger.info("="*80)
        logger.info("🔍 STAGE 2: EXTRACT FACTS")
        logger.info("="*80)

        # Initialize SpaCy if not provided
        if nlp_model is None:
            nlp_model = spacy.load("en_core_web_sm")

        doc = nlp_model(text)

        # Extract named entities
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]

        logger.info(f"✓ Named entities extracted: {len(entities)}")

        if debug and entities:
            # Group entities by type
            entity_types = {}
            for ent in entities:
                label = ent["label"]
                entity_types[label] = entity_types.get(label, 0) + 1

            logger.info(f"  → Entity types breakdown:")
            for label, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"     {label}: {count}")

            logger.info(f"  → Top 10 entities:")
            for i, ent in enumerate(entities[:10]):
                logger.info(f"     [{i+1}] {ent['text']} ({ent['label']})")

        # Extract key phrases (noun chunks)
        key_phrases = [
            chunk.text for chunk in doc.noun_chunks
            if len(chunk.text.split()) >= 2
        ]

        # Remove duplicates while preserving order
        key_phrases = list(dict.fromkeys(key_phrases))
        logger.info(f"✓ Key phrases extracted: {len(key_phrases)}")

        if debug and key_phrases:
            logger.info(f"  → Top 15 key phrases:")
            for i, phrase in enumerate(key_phrases[:15]):
                logger.info(f"     [{i+1}] {phrase}")

        return entities, key_phrases

    except Exception as e:
        logger.error(f"❌ Fact extraction failed: {str(e)}")
        raise


def create_text_processor(spacy_model: str = "en_core_web_sm") -> TextProcessor:
    """
    Factory function to create TextProcessor instance

    Args:
        spacy_model: Name of SpaCy model to use

    Returns:
        TextProcessor instance
    """
    return TextProcessor(spacy_model)
