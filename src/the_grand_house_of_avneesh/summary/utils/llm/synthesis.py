"""
LLM synthesis functions for creating comprehensive summaries
Builds prompts and processes LLM responses
"""

from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def build_synthesis_prompt(
    sentences: List[str],
    entities: List[Dict],
    key_phrases: List[str],
    target_words: int,
    max_entities: int = 20,
    max_phrases: int = 20
) -> Tuple[str, str]:
    """
    Build comprehensive prompt for LLM synthesis

    Args:
        sentences: Selected high-priority sentences
        entities: Extracted named entities
        key_phrases: Important themes and topics
        target_words: Target word count for summary
        max_entities: Maximum entities to include in prompt
        max_phrases: Maximum key phrases to include

    Returns:
        Tuple of (system_prompt, user_context)
    """
    # Prepare entity text
    entity_text = ", ".join([e["text"] for e in entities[:max_entities]]) if entities else "None"

    # Prepare key phrases
    key_phrase_text = ", ".join(key_phrases[:max_phrases]) if key_phrases else "None"

    # Format sentences with numbering
    sentences_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(sentences)])

    # System prompt
    system_prompt = """You are an expert content analyst and summarizer. Your task is to create comprehensive, well-structured summaries that capture ALL major themes and key information from the source material. Focus on breadth of coverage, not just depth in one area."""

    # User context with structured information
    user_context = f"""Analyze the following extracted information and create a comprehensive summary.

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

    return system_prompt, user_context


async def synthesize_summary(
    bedrock_client,
    sentences: List[str],
    entities: List[Dict],
    key_phrases: List[str],
    original_text: str,
    min_summary_words: int = 150,
    max_summary_words: int = 500,
    summary_target_ratio: float = 0.25,
    debug: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Stage 4: Synthesize comprehensive summary using LLM

    Args:
        bedrock_client: BedrockClient instance
        sentences: Selected high-priority sentences
        entities: Extracted named entities
        key_phrases: Important key phrases
        original_text: Original input text
        min_summary_words: Minimum words in summary
        max_summary_words: Maximum words in summary
        summary_target_ratio: Target ratio of summary to original (default 25%)
        debug: Enable debug logging

    Returns:
        Tuple of (summary_text, token_usage_dict)
    """
    try:
        logger.info("="*80)
        logger.info("🤖 STAGE 4: ENHANCED LLM SYNTHESIS")
        logger.info("="*80)

        # Calculate target word count
        original_words = len(original_text.split())
        target_words = max(
            min_summary_words,
            min(max_summary_words, int(original_words * summary_target_ratio))
        )

        logger.info(f"✓ Context prepared:")
        logger.info(f"  → Original text: ~{original_words} words")
        logger.info(f"  → Target summary: ~{target_words} words")
        logger.info(f"  → Entities used: {min(len(entities), 20)}")
        logger.info(f"  → Key phrases used: {min(len(key_phrases), 20)}")
        logger.info(f"  → Sentences used: {len(sentences)}")

        # Build prompt
        system_prompt, user_context = build_synthesis_prompt(
            sentences=sentences,
            entities=entities,
            key_phrases=key_phrases,
            target_words=target_words
        )

        prompt_chars = len(system_prompt) + len(user_context)
        logger.info(f"✓ Enhanced prompt size: {prompt_chars} characters (~{prompt_chars//4} tokens)")

        if debug:
            entity_preview = ", ".join([e["text"] for e in entities[:5]])
            phrase_preview = ", ".join(key_phrases[:5])
            logger.info(f"  → Entity preview: {entity_preview}...")
            logger.info(f"  → Theme preview: {phrase_preview}...")

        # Call LLM
        summary = await bedrock_client.invoke(
            system_prompt=system_prompt,
            user_prompt=user_context,
            debug=debug
        )

        if not summary:
            raise ValueError("Empty response from LLM")

        # Calculate token usage (estimated)
        completion_chars = len(summary)
        actual_words = len(summary.split())

        token_usage = {
            "prompt_tokens_estimated": prompt_chars // 4,
            "completion_tokens_estimated": completion_chars // 4,
            "total_tokens_estimated": (prompt_chars + completion_chars) // 4,
            "model": bedrock_client.model_id,
            "target_words": target_words,
            "actual_words": actual_words,
            "note": "Token counts are estimated (Bedrock doesn't provide exact counts)"
        }

        logger.info(f"✓ Summary generated:")
        logger.info(f"  → Length: {completion_chars} chars, {actual_words} words")
        logger.info(f"  → Target: {target_words} words")
        logger.info(f"  → Estimated tokens: {token_usage['total_tokens_estimated']}")

        return summary, token_usage

    except Exception as e:
        logger.error(f"❌ LLM synthesis failed: {str(e)}")
        raise


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text (rough approximation)

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def format_entity_list(entities: List[Dict], max_count: int = 20) -> str:
    """
    Format entity list for prompt

    Args:
        entities: List of entity dictionaries
        max_count: Maximum entities to include

    Returns:
        Formatted entity string
    """
    if not entities:
        return "None"
    return ", ".join([e["text"] for e in entities[:max_count]])


def format_key_phrases(key_phrases: List[str], max_count: int = 20) -> str:
    """
    Format key phrases for prompt

    Args:
        key_phrases: List of key phrases
        max_count: Maximum phrases to include

    Returns:
        Formatted phrase string
    """
    if not key_phrases:
        return "None"
    return ", ".join(key_phrases[:max_count])
