import os

from src.core.config import get_settings


def get_langfuse_handler(session_id: str | None = None):
    """Create a Langfuse callback handler when observability is explicitly enabled."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None

    if settings.langfuse_host and not os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host

    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            session_id=session_id,
        )
    except ImportError:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
