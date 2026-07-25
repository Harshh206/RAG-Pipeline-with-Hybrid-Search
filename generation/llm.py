from langchain_ollama import ChatOllama
from typing import Optional, List, Dict, Any
import logging
from config import config

logger = logging.getLogger(__name__)


class LMManager:
    """Simple LLM manager for Ollama"""

    def __init__(
        self,
        model_name: str = config.chat_model,
        base_url: str = config.base_url,
        temperature: float = config.temperature,
        **kwargs,
    ):
        """
        Initialize LLM manager

        Args:
            model_name: Ollama model name (default: llama3:8b)
            base_url: Ollama server URL
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        # Initialize the LLM
        self.llm = ChatOllama(
            model=model_name, base_url=base_url, temperature=temperature, **kwargs
        )
        logger.info(f"Initialized LLM: {model_name}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.invoke(messages)
            # Handle both string and list responses from ChatOllama
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                # Join list content into a single string
                return "".join(str(item) for item in response.content)
            else:
                return str(response.content)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def generate_with_context(self, query: str, context: str) -> str:
        """
        Generate response with context (for RAG)

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Generated response
        """
        system_prompt = """You are a helpful assistant. Answer questions based ONLY on the provided context.
If you cannot answer from the context, say "I don't have enough information to answer this."
Be concise and accurate."""

        prompt = f"""Context:{context}, Question: {query}
                    Answer based on the context above:"""

        return self.generate(prompt, system_prompt)

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Stream generate a response
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            for chunk in self.llm.stream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming: {e}")
            yield f"Error: {str(e)}"

    def validate(self) -> bool:
        """Test if LLM is working"""
        try:
            response = self.generate("Hello, respond with 'OK'")
            return "OK" in response or len(response) > 0
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return False
