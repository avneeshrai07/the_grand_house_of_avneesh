"""
Example usage of the modular TextSummarizer
"""

import asyncio
from text_summarizer import TextSummarizer
from utils.config.settings import SummarizerConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    """Example usage of the summarizer"""

    # Sample text to summarize
    sample_text = """
    Artificial intelligence (AI) has revolutionized numerous industries in recent years. 
    Machine learning algorithms now power everything from recommendation systems to autonomous vehicles.
    Natural language processing has enabled computers to understand and generate human language with 
    unprecedented accuracy. Deep learning networks, inspired by the human brain, have achieved 
    remarkable results in image recognition, speech synthesis, and game playing.

    The healthcare industry has particularly benefited from AI advancements. Medical imaging 
    systems can now detect diseases earlier and more accurately than ever before. Drug discovery 
    processes have been accelerated through AI-powered molecular modeling. Personalized treatment 
    plans are becoming possible through analysis of patient data and genetic information.

    However, these advances also raise important ethical questions. Privacy concerns emerge as 
    AI systems collect and analyze vast amounts of personal data. Bias in training data can lead 
    to discriminatory outcomes. The potential for job displacement creates economic uncertainties. 
    Researchers and policymakers are working to address these challenges while maximizing the 
    benefits of AI technology.
    """

    # Create custom configuration
    config = SummarizerConfig(
        score_threshold=0.7,
        min_sentences=3,
        max_sentences=10,
        temperature=0.3,
        debug=True
    )

    # Initialize summarizer
    summarizer = TextSummarizer(config)

    # Generate summary
    print("\nGenerating summary...\n")
    result = await summarizer.summarize(sample_text)

    # Display results
    print("\n" + "="*80)
    print("SUMMARY RESULT")
    print("="*80)
    print(result["summary"])
    print("\n" + "="*80)
    print("METRICS")
    print("="*80)
    for key, value in result["metrics"].items():
        print(f"{key}: {value}")
    print("\n" + "="*80)
    print("DEBUG INFO")
    print("="*80)
    for key, value in result["debug_info"].items():
        print(f"{key}: {value}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
