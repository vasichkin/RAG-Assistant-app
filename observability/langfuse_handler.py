"""
Langfuse callback handler factory.

Usage:
    from observability.langfuse_handler import get_handler

    handler = get_handler()
    answer = chain.invoke(query, config={"callbacks": [handler]})
    handler.flush()
"""

import functools

from langfuse.callback import CallbackHandler

from config.settings import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


@functools.lru_cache(maxsize=1)
def get_handler() -> CallbackHandler:
    """Return the shared Langfuse CallbackHandler (created once per process).

    Flush must be called by the caller after every chain invocation:
        handler.flush()
    """
    return CallbackHandler(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST,
    )
