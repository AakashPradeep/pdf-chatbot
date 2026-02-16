# config.py
"""
RESPONSIBILITY
--------------
This module is ONLY responsible for:
1) Loading environment variables from a .env file (optional path passed in)
2) Building a strongly-typed Settings object
3) Validating basic configuration constraints (e.g., overlap < chunk_size)
4) Initializing Chroma vector DB (embedded/persistent)
5) Initializing the LLM client (ChatOpenAI)

INPUTS
------
  # Precedence / priority order:
    # 1) OS/shell environment variables already set (highest priority)
    # 2) Values from the .env file (used only if the variable is NOT already set)
    #    - This is python-dotenv default: load_dotenv(..., override=False)
    # 3) Code defaults in os.getenv("KEY", "default") (lowest priority)
    #
    # If you want .env to override existing OS env vars, use:
    # load_dotenv(dotenv_path=dotenv_path, override=True)
    
- .env file (optional path)
- Environment variables (OPENAI_API_KEY, CHROMA_DIR, etc.)



OUTPUTS
-------
- Settings dataclass instance (load_settings)
- Chroma vector store instance (init_chroma)
- ChatOpenAI instance (init_llm)

EXPECTED EXCEPTIONS / FAILURE MODES
-----------------------------------
- ValueError: invalid numeric config (e.g., overlap >= chunk_size)
- Runtime/auth errors from OpenAI calls will show up later when embeddings/LLM is used,
  but missing OPENAI_API_KEY will usually fail when first OpenAI request is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# LangChain wrappers for OpenAI and Chroma vector store
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma


@dataclass(frozen=True)
class Settings:
    """
    A single source of truth for all configuration.

    Notes:
    - frozen=True makes the object immutable (safer; prevents accidental mutation).
    - We keep paths as Path objects (nicer than strings).
    """

    # Where Chroma will persist index files locally (embedded mode)
    chroma_dir: Path

    # Name of the collection inside Chroma
    collection_name: str

    # OpenAI model names
    embed_model: str
    chat_model: str

    # Chunking controls
    chunk_size: int
    chunk_overlap: int

    # Retrieval controls
    top_k: int

    # Prompt context size guardrail (avoid sending huge context to LLM)
    max_context_chars: int


def load_settings(dotenv_path: Optional[str] = None) -> Settings:
    """
    Read .env + environment variables and return a validated Settings object.

    INPUT
    -----
    dotenv_path: optional string path to .env
      - If provided: load_dotenv(dotenv_path=...)
      - If None: load_dotenv() looks for .env in current working directory

    OUTPUT
    ------
    Settings instance

    EXCEPTIONS
    ----------
    ValueError:
      - if CHUNK_OVERLAP >= CHUNK_SIZE (bad config; causes over-splitting)
      - if numeric env vars are not valid integers (int(...) fails)
    """

    # Load .env into environment variables, if present.
    # If dotenv_path is None, python-dotenv searches for ".env" by default.
    load_dotenv(dotenv_path=dotenv_path)

    # Read environment variables with defaults.
    # If you don’t set them in .env, these defaults are used.
    chroma_dir = Path(os.getenv("CHROMA_DIR", "./chroma_db")).expanduser().resolve()
    collection_name = os.getenv("CHROMA_COLLECTION", "pdf_rag")

    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

    # Convert numeric env vars from strings to integers.
    # If env var contains non-numeric value, int(...) will raise ValueError.
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k = int(os.getenv("TOP_K", "11"))
    max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "7000"))

    # Guardrail: overlap must be less than size.
    # If overlap is too high, step size collapses and chunk count explodes.
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Invalid config: CHUNK_OVERLAP ({chunk_overlap}) must be < CHUNK_SIZE ({chunk_size})."
        )

    return Settings(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embed_model=embed_model,
        chat_model=chat_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )


def init_chroma(settings: Settings) -> Chroma:
    """
    Initialize / load an embedded Chroma vector store with persistence.

    INPUT
    -----
    settings: Settings

    RESPONSIBILITY
    --------------
    - Create persist directory if missing
    - Create embeddings function (OpenAIEmbeddings)
    - Create Chroma instance pointing at persist_directory + collection_name

    OUTPUT
    ------
    Chroma vector store (ready for add_documents + retrieval)

    EXCEPTIONS / FAILURE MODES
    --------------------------
    - OSError/PermissionError: cannot create directory (bad permissions)
    - OpenAI auth/network errors: might show up when embeddings are actually computed
      (during add_documents), not necessarily here.
    """

    # Ensure local persistence folder exists.
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    # Embeddings function used by Chroma to embed documents and queries.
    # Requires OPENAI_API_KEY to be present in the environment.
    embeddings = OpenAIEmbeddings(model=settings.embed_model)

    # Create/load Chroma collection (embedded mode).
    # If files already exist in persist_directory, Chroma will reuse them.
    vs = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
    )
    return vs


def init_llm(settings: Settings) -> ChatOpenAI:
    """
    Initialize LLM client used for final answer generation.

    INPUT
    -----
    settings: Settings

    OUTPUT
    ------
    ChatOpenAI instance (LangChain wrapper)

    FAILURE MODES
    -------------
    - Missing/invalid OPENAI_API_KEY typically fails once you call the model (not here).
    """
    return ChatOpenAI(model=settings.chat_model, temperature=0)