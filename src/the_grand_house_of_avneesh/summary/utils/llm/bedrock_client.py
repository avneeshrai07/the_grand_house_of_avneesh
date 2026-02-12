"""
AWS Bedrock LLM client wrapper
Handles initialization and async invocation
"""

from langchain_aws import ChatBedrock
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BedrockClient:
    """Wrapper for AWS Bedrock LLM client"""

    def __init__(
        self,
        model_id: str = "amazon.nova-lite-v1:0",
        region_name: str = "ap-southeast-2",
        temperature: float = 0.3,
        max_tokens: int = 3096
    ):
        """
        Initialize Bedrock client

        Args:
            model_id: AWS Bedrock model ID
            region_name: AWS region
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum tokens for response
        """
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            logger.info("☁️ Initializing AWS Bedrock client...")
            logger.info(f"  → Model: {self.model_id}")
            logger.info(f"  → Region: {self.region_name}")
            logger.info(f"  → Temperature: {self.temperature}")
            logger.info(f"  → Max Tokens: {self.max_tokens}")

            self.llm = ChatBedrock(
                model_id=self.model_id,
                region_name=self.region_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.info(f"✅ Bedrock client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bedrock: {str(e)}")
            raise

    async def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        debug: bool = False
    ) -> str:
        """
        Async invoke Bedrock LLM

        Args:
            system_prompt: System instructions
            user_prompt: User query/context
            debug: Enable debug logging

        Returns:
            LLM response text
        """
        try:
            if debug:
                logger.info("\n" + "="*80)
                logger.info("🤖 CALLING BEDROCK LLM")
                logger.info("="*80)
                logger.info(f"System Prompt: {system_prompt[:100]}...")
                logger.info(f"User Prompt Length: {len(user_prompt)} chars")
                logger.info(f"User Prompt Preview: {user_prompt[:200]}...")

            response = await self.llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            if response is None:
                raise ValueError("Received None response from Bedrock")

            # Extract content
            if hasattr(response, "content") and isinstance(response.content, list):
                text = "".join(
                    block.get("text", "")
                    for block in response.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
            elif hasattr(response, "content"):
                text = str(response.content).strip()
            else:
                text = str(response).strip()

            if debug:
                logger.info(f"✅ LLM Response Length: {len(text)} chars")
                logger.info(f"Response Preview: {text[:400]}...")
                logger.info("="*80)

            return text

        except Exception as e:
            logger.error(f"❌ Critical error in Bedrock invocation: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """Get current client configuration"""
        return {
            "model_id": self.model_id,
            "region_name": self.region_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


async def create_bedrock_client(
    model_id: str = "amazon.nova-lite-v1:0",
    region_name: str = "ap-southeast-2",
    temperature: float = 0.3,
    max_tokens: int = 3096
) -> BedrockClient:
    """
    Factory function to create BedrockClient instance

    Args:
        model_id: AWS Bedrock model ID
        region_name: AWS region
        temperature: LLM temperature
        max_tokens: Max tokens for response

    Returns:
        BedrockClient instance
    """
    return BedrockClient(
        model_id=model_id,
        region_name=region_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
