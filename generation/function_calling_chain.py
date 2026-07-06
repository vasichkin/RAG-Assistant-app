"""
Agentic RAG chain using Gemini function calling + structured JSON output.

Usage:
    from generation.function_calling_chain import run_agentic_rag
    result = run_agentic_rag("What is IFC's net income for FY24?", retriever)
    # result == {"answer": "...", "page_references": [12, 45]}
"""

import json
import logging
from typing import Any

from google.genai import types
from pydantic import BaseModel

from config.settings import client, GENERATION_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

_SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_ifc_report",
            description=(
                "Search the IFC 2024 Annual Report for relevant text, tables, "
                "and image descriptions. Call this whenever you need facts from "
                "the report to answer the user's question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    }
                },
                "required": ["query"],
            },
        )
    ]
)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class FinancialAnswer(BaseModel):
    answer: str
    page_references: list[int]


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agentic_rag(
    user_query: str,
    retriever: Any,
    max_turns: int = 3,
) -> dict:
    """Run Gemini with function calling against a LangChain retriever.

    Gemini decides when to call ``search_ifc_report``; we execute the retriever
    and feed results back.  After the tool loop, a second call requests a
    structured JSON answer conforming to ``FinancialAnswer``.

    Args:
        user_query: The end-user question.
        retriever:  Any LangChain BaseRetriever (.invoke returns list[Document]).
        max_turns:  Maximum tool-call rounds before forcing a final answer.

    Returns:
        dict with keys ``answer`` (str) and ``page_references`` (list[int]).
    """
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    ]

    for turn in range(max_turns):
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(tools=[_SEARCH_TOOL]),
        )

        candidate_content = response.candidates[0].content
        # Always record the model's turn before checking for tool calls
        messages.append(candidate_content)

        fc_parts = [p for p in candidate_content.parts if p.function_call]

        if not fc_parts:
            # Gemini decided it has enough context — exit the loop
            break

        # Execute every function call in this turn
        fn_response_parts = []
        for part in fc_parts:
            fc = part.function_call
            logger.debug("Function call: %s(%s)", fc.name, fc.args)
            try:
                docs = retriever.invoke(fc.args["query"])
                context = "\n\n---\n\n".join(
                    f"[Page {d.metadata.get('page_number', '?')}]\n{d.page_content}"
                    for d in docs[:5]
                )
            except Exception as exc:
                logger.warning("Retriever error: %s", exc)
                context = "No results found."

            fn_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"content": context},
                )
            )

        messages.append(types.Content(role="user", parts=fn_response_parts))

    # If the last message is a user turn (fn_response_parts from loop exhaustion),
    # get model acknowledgement first to maintain strict user/model alternation.
    if messages and getattr(messages[-1], "role", None) == "user":
        ack_response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(tools=[_SEARCH_TOOL]),
        )
        messages.append(ack_response.candidates[0].content)

    # Request structured final answer
    messages.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text="Provide your final answer using the search results above. "
                "Include the page numbers you used."
            )],
        )
    )
    final = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=messages,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FinancialAnswer,
        ),
    )
    try:
        return json.loads(final.text)
    except (json.JSONDecodeError, TypeError):
        return {"answer": final.text, "page_references": []}
