# Text Summarization Utils

Modular NLP-based text summarization utilities with AWS Bedrock LLM integration.

## Directory Structure

```
utils/
├── __init__.py                 # Package initialization
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration management
├── nlp/
│   ├── __init__.py
│   ├── text_processor.py      # Text parsing and fact extraction
│   └── sentence_ranker.py     # TF-IDF based sentence ranking
├── llm/
│   ├── __init__.py
│   ├── bedrock_client.py      # AWS Bedrock client wrapper
│   └── synthesis.py           # LLM summary synthesis
├── metrics/
│   ├── __init__.py
│   └── calculator.py          # Metrics calculation
└── exceptions.py              # Custom exceptions
```

## Features

### 4-Stage Summarization Pipeline

1. **Parse Structure** - Rule-based document parsing into paragraphs and sentences
2. **Extract Facts** - NLP-based entity and key phrase extraction using SpaCy
3. **Rank & Filter** - Adaptive TF-IDF sentence selection with dynamic thresholding
4. **LLM Synthesis** - Comprehensive summary generation using AWS Bedrock

### Key Capabilities

- ✅ Adaptive sentence selection (threshold-based, not fixed count)
- ✅ Configurable parameters via `SummarizerConfig`
- ✅ Comprehensive metrics (compression ratio, word/char counts)
- ✅ Detailed debug logging
- ✅ Async/await support for LLM calls
- ✅ Modular architecture for easy testing and extension

## Installation

```bash
# Install dependencies
pip install spacy scikit-learn langchain-aws boto3 python-dotenv

# Download SpaCy model
python -m spacy download en_core_web_sm
```

## Environment Variables

Create a `.env` file:

```env
MODEL_ID=amazon.nova-lite-v1:0
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
TEMPERATURE=0.3
MAX_TOKENS=3096
DEBUG=true
```

## Usage

### Basic Usage

```python
import asyncio
from text_summarizer import TextSummarizer
from utils.config.settings import SummarizerConfig

async def main():
    # Create configuration
    config = SummarizerConfig(
        score_threshold=0.7,
        min_sentences=3,
        max_sentences=15,
        temperature=0.3,
        debug=True
    )

    # Initialize summarizer
    summarizer = TextSummarizer(config)

    # Summarize text
    text = "Your long text here..."
    result = await summarizer.summarize(text)

    print(result["summary"])
    print(result["metrics"])

asyncio.run(main())
```

### Using Individual Utilities

```python
from utils.nlp.text_processor import parse_structure, extract_facts
from utils.nlp.sentence_ranker import rank_and_filter_sentences
from utils.metrics.calculator import calculate_metrics

# Parse structure
paragraphs, sentences = await parse_structure(text)

# Extract facts
entities, key_phrases = await extract_facts(text)

# Rank sentences
top_sentences = await rank_and_filter_sentences(
    sentences,
    score_threshold=0.7,
    min_sentences=3,
    max_sentences=15
)

# Calculate metrics
metrics = calculate_metrics(original_text, summary_text)
```

### Custom Configuration

```python
from utils.config.settings import SummarizerConfig

# Create custom config
config = SummarizerConfig(
    score_threshold=0.8,          # Higher threshold = fewer sentences
    min_sentences=5,              # Minimum sentences to include
    max_sentences=20,             # Maximum sentences to include
    model_id="amazon.nova-pro-v1:0",
    temperature=0.2,              # Lower = more focused
    max_tokens=4096,
    debug=False
)

# Or load from environment
config = SummarizerConfig.from_env()
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `score_threshold` | 0.7 | TF-IDF score threshold (0.0-1.0) |
| `min_sentences` | 3 | Minimum sentences to select |
| `max_sentences` | 15 | Maximum sentences to select |
| `model_id` | amazon.nova-lite-v1:0 | AWS Bedrock model |
| `region_name` | ap-southeast-2 | AWS region |
| `temperature` | 0.3 | LLM temperature |
| `max_tokens` | 3096 | Max tokens for LLM response |
| `debug` | True | Enable debug logging |

## Output Format

```python
{
    "summary": "Generated summary text...",
    "metrics": {
        "original_chars": 5000,
        "compressed_chars": 800,
        "original_words": 850,
        "compressed_words": 150,
        "char_reduction_percentage": "84.0%",
        "compression_ratio": "0.16x",
        ...
    },
    "token_usage": {
        "prompt_tokens_estimated": 400,
        "completion_tokens_estimated": 200,
        "total_tokens_estimated": 600,
        "model": "amazon.nova-lite-v1:0",
        ...
    },
    "debug_info": {
        "total_paragraphs": 5,
        "total_sentences": 25,
        "selected_sentences": 8,
        "total_entities": 15,
        "total_key_phrases": 20,
        ...
    }
}
```

## Error Handling

```python
from utils.exceptions import (
    SummarizationError,
    NLPProcessingError,
    LLMError
)

try:
    result = await summarizer.summarize(text)
except NLPProcessingError as e:
    print(f"NLP processing failed: {e}")
except LLMError as e:
    print(f"LLM call failed: {e}")
except SummarizationError as e:
    print(f"Summarization failed: {e}")
```

## Testing

```python
# Test individual components
import asyncio
from utils.nlp.text_processor import parse_structure

async def test_parsing():
    text = "Sample text. Another sentence."
    paragraphs, sentences = await parse_structure(text, debug=True)
    assert len(sentences) == 2

asyncio.run(test_parsing())
```

## Architecture

The system uses a modular pipeline architecture:

1. **Configuration Layer** (`utils/config/`) - Centralized settings management
2. **NLP Layer** (`utils/nlp/`) - Text processing and analysis
3. **LLM Layer** (`utils/llm/`) - Language model integration
4. **Metrics Layer** (`utils/metrics/`) - Performance measurement
5. **Orchestration Layer** (`text_summarizer.py`) - Pipeline coordination

## License

MIT License

## Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- Add tests for new features
- Update documentation
- Follow PEP 8 style guidelines
