"""
Text Summarizer Module with Dynamic Sentence Selection
Implements the 4-stage NLP pipeline with intelligent adaptive sentence filtering
"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Any
import asyncio
from langchain_aws import ChatBedrock
import logging
from datetime import datetime
import json
import numpy as np

from .exceptions import SummarizationError, NLPProcessingError
from dotenv import load_dotenv
load_dotenv()
import os

# Configure logging with detailed format
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

    def __init__(
        self,
        score_threshold: float = 0.7,  # Adaptive threshold instead of fixed count
        min_sentences: int = 3,
        max_sentences: int = 15,
        model_id: str = os.getenv("MODEL_ID", "amazon.nova-lite-v1:0"),
        region_name: str = os.getenv("AWS_REGION", "ap-southeast-2"),
        temperature: float = 0.3,  # Increased for more creative output
        max_tokens: int = 3096,  # Increased for longer summaries
        debug: bool = True
    ):
        """
        Initialize the summarizer with adaptive sentence selection

        Args:
            score_threshold: Relative threshold for sentence selection (0.0-1.0)
            min_sentences: Minimum sentences to include
            max_sentences: Maximum sentences to include
            model_id: AWS Bedrock model ID
            region_name: AWS region
            temperature: LLM temperature (0.3 for balanced creativity)
            max_tokens: Maximum tokens for LLM response
            debug: Enable detailed debugging output
        """
        self.score_threshold = score_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug

        logger.info("=" * 80)
        logger.info("🚀 INITIALIZING ADAPTIVE TEXT SUMMARIZER")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_id}")
        logger.info(f"Region: {self.region_name}")
        logger.info(f"Score Threshold: {self.score_threshold} (adaptive)")
        logger.info(f"Sentence Range: {self.min_sentences}-{self.max_sentences}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Max Tokens: {self.max_tokens}")
        logger.info(f"Debug Mode: {self.debug}")

        # Initialize SpaCy
        try:
            logger.info("📚 Loading SpaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ SpaCy model loaded successfully")
        except OSError:
            raise NLPProcessingError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Install it with: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
            )

        # Initialize AWS Bedrock
        try:
            logger.info("☁️ Initializing AWS Bedrock client...")
            self.llm = ChatBedrock(
                model_id=self.model_id,
                region_name=self.region_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.info(f"✅ Bedrock client initialized: {self.model_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bedrock: {str(e)}")
            raise SummarizationError(f"Failed to initialize AWS Bedrock client: {str(e)}")

        logger.info("=" * 80)

    async def simple_bedrock_llm(self, system_prompt: str, context: str) -> str:
        """Async function to call Bedrock with detailed logging"""
        try:
            if self.debug:
                logger.info("\n" + "="*80)
                logger.info("🤖 CALLING BEDROCK LLM")
                logger.info("="*80)
                logger.info(f"System Prompt: {system_prompt[:100]}...")
                logger.info(f"Context Length: {len(context)} chars")
                logger.info(f"Context Preview: {context[:300]}...")

            response = await self.llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ])

            if response is None:
                raise SummarizationError("Received None response from Bedrock")

            # Extract content
            if hasattr(response, "content") and isinstance(response.content, list):
                summary = "".join(
                    block.get("text", "")
                    for block in response.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
            elif hasattr(response, "content"):
                summary = str(response.content).strip()
            else:
                summary = str(response).strip()

            if self.debug:
                logger.info(f"✅ LLM Response Length: {len(summary)} chars")
                logger.info(f"Response Preview: {summary[:400]}...")
                logger.info("="*80)

            return summary

        except Exception as e:
            logger.critical(f"❌ Critical error in simple_bedrock_llm: {e}")
            raise SummarizationError(f"Bedrock LLM call failed: {str(e)}")

    async def summarize(self, text: str) -> Dict[str, Any]:
        """Main summarization pipeline with step-by-step debugging"""

        logger.info("\n" + "🔥" * 40)
        logger.info("STARTING ADAPTIVE SUMMARIZATION PIPELINE")
        logger.info("🔥" * 40)
        logger.info(f"Input Text Length: {len(text)} characters")
        logger.info(f"Input Text Preview: {text[:200]}...")

        try:
            # Stage 1: Parse Structure
            logger.info("\n" + "─"*80)
            logger.info("📝 STAGE 1: PARSE STRUCTURE (Rule-based, No LLM)")
            logger.info("─"*80)
            paragraphs, sentences = await self._parse_structure(text)

            # Stage 2: Extract Facts
            logger.info("\n" + "─"*80)
            logger.info("🔍 STAGE 2: EXTRACT FACTS (NLP/SpaCy, No LLM)")
            logger.info("─"*80)
            entities, key_phrases = await self._extract_facts(text)

            # Stage 3: Adaptive Rank & Filter
            logger.info("\n" + "─"*80)
            logger.info("📊 STAGE 3: ADAPTIVE RANK & FILTER (TF-IDF + Threshold)")
            logger.info("─"*80)
            top_sentences = await self._rank_and_filter(sentences)

            # Stage 4: Enhanced LLM Synthesis
            logger.info("\n" + "─"*80)
            logger.info("🤖 STAGE 4: ENHANCED LLM SYNTHESIS (ONE Bedrock Call)")
            logger.info("─"*80)
            summary, token_usage = await self._llm_synthesis(
                top_sentences,
                entities,
                key_phrases,
                text
            )

            # Calculate metrics
            logger.info("\n" + "─"*80)
            logger.info("📐 CALCULATING METRICS")
            logger.info("─"*80)
            metrics = self._calculate_metrics(text, summary)

            # Final summary
            logger.info("\n" + "🎯" * 40)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("🎯" * 40)
            logger.info(f"Original: {metrics['original_chars']} chars")
            logger.info(f"Summary: {metrics['compressed_chars']} chars")
            logger.info(f"Reduction: {metrics['reduction_percentage']}")
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
                    "selection_method": "adaptive_threshold"
                }
            }

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise SummarizationError(f"Summarization pipeline failed: {str(e)}")

    async def _parse_structure(self, text: str) -> Tuple[List[str], List[str]]:
        """Stage 1: Parse document structure with debugging"""
        try:
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            logger.info(f"✓ Paragraphs extracted: {len(paragraphs)}")

            if self.debug and paragraphs:
                logger.info(f"  → First paragraph: {paragraphs[0][:100]}...")
                if len(paragraphs) > 1:
                    logger.info(f"  → Last paragraph: {paragraphs[-1][:100]}...")

            # Split into sentences using SpaCy
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            logger.info(f"✓ Sentences extracted: {len(sentences)}")

            if self.debug and sentences:
                logger.info(f"  → Sample sentences (first 3):")
                for i, sent in enumerate(sentences[:3]):
                    logger.info(f"     [{i+1}] {sent[:100]}...")

            # Sentence length analysis
            avg_sent_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
            logger.info(f"✓ Average sentence length: {avg_sent_length:.1f} chars")

            return paragraphs, sentences

        except Exception as e:
            logger.error(f"❌ Structure parsing failed: {str(e)}")
            raise NLPProcessingError(f"Structure parsing failed: {str(e)}")

    async def _extract_facts(self, text: str) -> Tuple[List[Dict], List[str]]:
        """Stage 2: Extract facts with detailed entity analysis"""
        try:
            doc = self.nlp(text)

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

            if self.debug and entities:
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

            # Remove duplicates
            key_phrases = list(dict.fromkeys(key_phrases))
            logger.info(f"✓ Key phrases extracted: {len(key_phrases)}")

            if self.debug and key_phrases:
                logger.info(f"  → Top 15 key phrases:")
                for i, phrase in enumerate(key_phrases[:15]):
                    logger.info(f"     [{i+1}] {phrase}")

            return entities, key_phrases

        except Exception as e:
            logger.error(f"❌ Fact extraction failed: {str(e)}")
            raise NLPProcessingError(f"Fact extraction failed: {str(e)}")

    async def _rank_and_filter(self, sentences: List[str]) -> List[str]:
        """Stage 3: Adaptive sentence ranking - selects best sentences dynamically"""
        try:
            logger.info(f"✓ Input sentences: {len(sentences)}")
            logger.info(f"✓ Selection strategy: Adaptive threshold-based")
            logger.info(f"✓ Range: {self.min_sentences}-{self.max_sentences} sentences")

            if len(sentences) <= self.min_sentences:
                logger.info("  → All sentences included (below minimum)")
                return sentences

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=100,
                ngram_range=(1, 2)  # Include bigrams for better context
            )

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            logger.info(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

            # Calculate scores
            sentence_scores = tfidf_matrix.sum(axis=1).A1

            # Normalize scores to 0-1 range
            if sentence_scores.max() > 0:
                normalized_scores = (sentence_scores - sentence_scores.min()) / (sentence_scores.max() - sentence_scores.min())
            else:
                normalized_scores = sentence_scores

            logger.info(f"✓ Sentence scores computed and normalized")

            if self.debug:
                logger.info(f"  → Raw score statistics:")
                logger.info(f"     Min: {sentence_scores.min():.4f}")
                logger.info(f"     Max: {sentence_scores.max():.4f}")
                logger.info(f"     Mean: {sentence_scores.mean():.4f}")
                logger.info(f"     Median: {np.median(sentence_scores):.4f}")
                logger.info(f"     Std: {sentence_scores.std():.4f}")

            # Adaptive threshold: select sentences above threshold
            threshold_value = self.score_threshold
            selected_indices = np.where(normalized_scores >= threshold_value)[0]

            # Enforce min/max constraints
            if len(selected_indices) < self.min_sentences:
                # Select top N if below minimum
                top_indices = sentence_scores.argsort()[-self.min_sentences:][::-1]
                selected_indices = sorted(top_indices)
                logger.info(f"  ⚠️ Below minimum, selected top {self.min_sentences} sentences")
            elif len(selected_indices) > self.max_sentences:
                # Take only top N if above maximum
                top_from_selected = sentence_scores[selected_indices].argsort()[-self.max_sentences:][::-1]
                selected_indices = sorted(selected_indices[top_from_selected])
                logger.info(f"  ⚠️ Above maximum, limited to top {self.max_sentences} sentences")
            else:
                selected_indices = sorted(selected_indices)
                logger.info(f"  ✓ Selected {len(selected_indices)} sentences above threshold {threshold_value}")

            top_sentences = [sentences[i] for i in selected_indices]

            if self.debug:
                logger.info(f"  → Selected {len(selected_indices)} sentences:")
                for i, idx in enumerate(selected_indices[:5]):
                    score = sentence_scores[idx]
                    norm_score = normalized_scores[idx]
                    logger.info(f"     [{i+1}] Score: {score:.4f} (norm: {norm_score:.2f}) | {sentences[idx][:70]}...")

            return top_sentences

        except Exception as e:
            logger.error(f"❌ Ranking failed: {str(e)}")
            raise NLPProcessingError(f"Ranking and filtering failed: {str(e)}")

    async def _llm_synthesis(
        self,
        sentences: List[str],
        entities: List[Dict],
        key_phrases: List[str],
        original_text: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Stage 4: Enhanced LLM synthesis with comprehensive prompt"""
        try:
            # Prepare context
            entity_text = ", ".join([e["text"] for e in entities[:20]]) if entities else "None"
            key_phrase_text = ", ".join(key_phrases[:20]) if key_phrases else "None"
            sentences_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(sentences)])

            # Estimate word count for summary (aim for 15-25% of original)
            original_words = len(original_text.split())
            target_words = max(150, min(500, int(original_words * 0.25)))

            logger.info(f"✓ Context prepared:")
            logger.info(f"  → Original text: ~{original_words} words")
            logger.info(f"  → Target summary: ~{target_words} words")
            logger.info(f"  → Entities used: {len(entities[:20])}")
            logger.info(f"  → Key phrases used: {len(key_phrases[:20])}")
            logger.info(f"  → Sentences used: {len(sentences)}")

            # Enhanced system prompt
            system_prompt = """You are an expert content analyst and summarizer. Your task is to create comprehensive, well-structured summaries that capture ALL major themes and key information from the source material. Focus on breadth of coverage, not just depth in one area."""

            # Comprehensive user prompt with clear instructions
            context = f"""Analyze the following extracted information and create a comprehensive summary.

📊 EXTRACTED KEY ENTITIES:
{entity_text}

🔑 IMPORTANT THEMES AND TOPICS:
{key_phrase_text}

📝 HIGH-PRIORITY SENTENCES (in document order):
{sentences_text}

🎯 INSTRUCTIONS:
1. Create a comprehensive summary of approximately {target_words} words
2. Cover ALL major topics represented in the key phrases and entities
3. Maintain chronological flow when relevant
4. Include specific names, dates, numbers, and concrete details from entities
5. Synthesize information - don't just concatenate sentences
6. Write in clear, professional prose with smooth transitions
7. Ensure the summary is self-contained and readable without the original text

⚠️ CRITICAL: Your summary must cover the FULL SCOPE of topics present in the material, not just the first few sentences. Look for diversity of themes."""

            prompt_chars = len(system_prompt) + len(context)
            logger.info(f"✓ Enhanced prompt size: {prompt_chars} characters (~{prompt_chars//4} tokens)")

            if self.debug:
                logger.info(f"  → Entity preview: {entity_text[:150]}...")
                logger.info(f"  → Theme preview: {key_phrase_text[:150]}...")

            # Call LLM
            summary = await self.simple_bedrock_llm(system_prompt, context)

            if not summary:
                raise SummarizationError("Empty response from Bedrock")

            # Calculate token usage
            completion_chars = len(summary)

            token_usage = {
                "prompt_tokens_estimated": prompt_chars // 4,
                "completion_tokens_estimated": completion_chars // 4,
                "total_tokens_estimated": (prompt_chars + completion_chars) // 4,
                "model": self.model_id,
                "target_words": target_words,
                "actual_words": len(summary.split()),
                "note": "Token counts are estimated (Bedrock doesn't provide exact counts)"
            }

            logger.info(f"✓ Summary generated:")
            logger.info(f"  → Length: {completion_chars} chars, {token_usage['actual_words']} words")
            logger.info(f"  → Target: {target_words} words")
            logger.info(f"  → Estimated tokens: {token_usage['total_tokens_estimated']}")

            return summary, token_usage

        except Exception as e:
            logger.error(f"❌ LLM synthesis failed: {str(e)}")
            raise SummarizationError(f"LLM synthesis failed: {str(e)}")

    def _calculate_metrics(self, original: str, summary: str) -> Dict[str, Any]:
        """Calculate compression metrics with logging"""
        original_chars = len(original)
        original_words = len(original.split())
        compressed_chars = len(summary)
        compressed_words = len(summary.split())

        reduction_percentage = ((original_chars - compressed_chars) / original_chars) * 100
        compression_ratio = compressed_chars / original_chars

        metrics = {
            "original_chars": original_chars,
            "original_words": original_words,
            "compressed_chars": compressed_chars,
            "compressed_words": compressed_words,
            "reduction_percentage": f"{reduction_percentage:.1f}%",
            "compression_ratio": f"{compression_ratio:.2f}x"
        }

        logger.info(f"✓ Metrics calculated:")
        logger.info(f"  → Original: {original_chars:,} chars, {original_words} words")
        logger.info(f"  → Summary: {compressed_chars:,} chars, {compressed_words} words")
        logger.info(f"  → Reduction: {reduction_percentage:.1f}%")
        logger.info(f"  → Ratio: {compression_ratio:.2f}x")

        return metrics
