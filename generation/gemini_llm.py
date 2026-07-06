"""
Custom LangChain BaseChatModel wrapping the Vertex AI Gemini generate_content API.

Usage:
    from generation.gemini_llm import GeminiChatModel
    llm = GeminiChatModel()
"""

from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from config.settings import GENERATION_MODEL, client


class GeminiChatModel(BaseChatModel):
    """LangChain BaseChatModel backed by Vertex AI Gemini via google-genai SDK."""

    model_name: str = GENERATION_MODEL

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _convert_messages(self, messages: List) -> List[dict]:
        """Convert LangChain messages to google.genai content format.

        SystemMessage content is prepended to the first user message.
        """
        system_parts: List[str] = []
        genai_messages: List[dict] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append(msg.content)
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if system_parts:
                    content = "\n\n".join(system_parts) + "\n\n" + content
                    system_parts = []
                genai_messages.append({"role": "user", "parts": [{"text": content}]})
            elif isinstance(msg, AIMessage):
                genai_messages.append({"role": "model", "parts": [{"text": msg.content}]})
            else:
                # Fallback: treat unknown message types as user messages
                genai_messages.append({"role": "user", "parts": [{"text": str(msg.content)}]})

        # If system messages were never merged (no human message followed), prepend as user turn
        if system_parts:
            genai_messages.insert(0, {"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]})

        return genai_messages

    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        genai_messages = self._convert_messages(messages)
        gen_cfg = {"stop_sequences": stop} if stop else None
        response = client.models.generate_content(
            model=self.model_name,
            contents=genai_messages,
            config=gen_cfg,
        )
        text = response.text
        ai_message = AIMessage(content=text)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        genai_messages = self._convert_messages(messages)
        gen_cfg = {"stop_sequences": stop} if stop else {}
        for chunk in client.models.generate_content_stream(
            model=self.model_name,
            contents=genai_messages,
            config=gen_cfg or None,
        ):
            if chunk.text:
                token_chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk.text))
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=token_chunk)
                yield token_chunk
