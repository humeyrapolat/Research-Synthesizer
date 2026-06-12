import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    uvicorn_reload: bool
    groq_api_key: str | None
    tavily_api_key: str | None
    langfuse_enabled: bool
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str | None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        uvicorn_reload=_env_bool("UVICORN_RELOAD"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        langfuse_enabled=_env_bool("LANGFUSE_ENABLED"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST"),
    )
