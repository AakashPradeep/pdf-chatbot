# app.py
"""
UPDATED AGAIN: Adds proper try/except handling in key places.

GOALS
-----
1) Never crash the whole ingest run because ONE PDF is bad.
2) Print a clear error message with the failing file and the stage.
3) Keep checkpoint updated so reruns don't redo already-successful work.
4) Fail fast only for truly fatal issues (e.g., cannot load settings).

NOTES ON EXCEPTION STRATEGY
---------------------------
- Ingest:
  - Per-PDF errors are caught and logged; we continue with next file.
  - We wrap "read PDF", "chunk", "add_documents", "delete vectors", "persist", "checkpoint write"
    to pinpoint failures.
- Chat:
  - Retrieval errors and LLM errors are caught; we print a friendly message and keep the loop alive.

If you want, you can later add retries for OpenAI rate limit errors.
PDF Q&A CLI

Two commands:
- ingest: scan PDFs, chunk, embed, store in Chroma (incremental via checkpoint)
- chat: ask questions; retrieve context from Chroma and stream answers

Design goals:
- Ingest should NOT crash on a single bad PDF.
- Chat should keep going even if a single retrieval/LLM call fails.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, List

import click

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

from .config import load_settings, init_chroma, init_llm
from .session import SessionStats, usage_from_metadata, TurnUsage
from langchain_core.documents import Document
from .pdf_pipeline import extract_all_units
from .SummaryBasedMemory import SummaryBasedMemory

# -----------------------------
# Checkpoint helpers
# -----------------------------
def checkpoint_path(chroma_dir: Path) -> Path:
    """Single checkpoint file storing state for ALL PDFs."""
    return chroma_dir / ".checkpoint.json"


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        return {"version": 1, "files": {}}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        click.echo(f"⚠️  Checkpoint file is unreadable ({path}): {e}")
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            path.rename(bak)
            click.echo(f"⚠️  Renamed corrupt checkpoint to: {bak}")
        except Exception as re:
            click.echo(f"⚠️  Could not rename corrupt checkpoint: {re}")
        return {"version": 1, "files": {}}


def save_checkpoint(path: Path, state: Dict) -> None:
    """Atomic-ish write to reduce corruption risk."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        click.echo(f"⚠️  Failed to write checkpoint {path}: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def classify_pdf(pdf_path: Path, state: Dict) -> Tuple[str, str]:
    """
    Returns: (status, sha256)
    status in: new | changed | unchanged
    """
    h = sha256_file(pdf_path)
    key = str(pdf_path)
    old = state["files"].get(key)
    if not old:
        return "new", h
    if old.get("sha256") != h:
        return "changed", h
    return "unchanged", h


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -----------------------------
# PDF discovery + ingestion
# -----------------------------

def iter_pdf_paths(pdf_root: Path) -> Iterator[Path]:
    for p in pdf_root.rglob("*.pdf"):
        if p.is_file():
            yield p


def ingest_one_pdf(pdf_path: Path, splitter: RecursiveCharacterTextSplitter):
    """
    PDF -> content units (text/table/image OCR) -> Documents -> chunk
    """
    units = extract_all_units(
        pdf_path,
        include_text=True,
        include_tables=True,
        include_images=True,
        ocr_images=True,
    )

    docs = []
    for u in units:
        docs.append(Document(page_content=u.text, metadata=u.metadata))

    # chunk everything (even tables/images) so big blocks don’t blow context window
    chunks = splitter.split_documents(docs)
    
    print(f"📄 {pdf_path.name}: extracted {len(units)} units -> {len(chunks)} chunks")
    return chunks


def delete_vectors_for_pdf(vs, pdf_path: Path) -> None:
    """Delete stored vectors for a PDF using metadata filter."""
    vs._collection.delete(where={"source": str(pdf_path)})


# -----------------------------
# Retrieval context formatting
# -----------------------------

def format_docs(docs, max_chars: int) -> str:
    parts: List[str] = []
    total = 0

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)

        header = f"[source={Path(src).name}"
        if isinstance(page, int):
            header += f" page={page + 1}"
        header += "]"

        text = (d.page_content or "").strip()
        if not text:
            continue

        block = header + "\n" + text + "\n"
        if total + len(block) > max_chars:
            break

        parts.append(block)
        parts.append(f"\n today date is :{now_utc_iso()}")  
        total += len(block)

    return "\n---\n".join(parts)


def retrieve_docs(retriever, query: str):
    """Compatibility helper across LangChain versions."""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


# -----------------------------
# Chat input helpers
# -----------------------------

def read_question(single_line: bool = True) -> str:
    """
    If single_line=True: prompt once.
    If single_line=False: read until a line containing only '--'.
    """
    if single_line:
        return click.prompt("You", default="", show_default=False).strip()

    click.echo("Type multi-line question. End with a line containing only: --")
    lines = []
    while True:
        line = click.prompt(">", default="", show_default=False)
        if line.strip() == "--" or line.strip() == "\n\n":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def print_user_bubble(q: str) -> None:
    click.echo("\n" + "-" * 70)
    click.echo("You:")
    click.echo(q)
    click.echo("-" * 70)


# -----------------------------
# Click CLI
# -----------------------------

@click.group()
@click.option("--env-file", default=None, help="Path to .env file (optional).")
@click.pass_context
def cli(ctx: click.Context, env_file: Optional[str]):
    """CLI entrypoint (sets up settings + chroma + llm)."""
    try:
        settings = load_settings(dotenv_path=env_file)
        vs = init_chroma(settings)
        llm = init_llm(settings)
    except Exception as e:
        raise click.ClickException(f"Failed to initialize app: {e}")

    ctx.obj = {"settings": settings, "vs": vs, "llm": llm}


@cli.command()
@click.option(
    "--pdf-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root folder containing PDFs (recursive scan).",
)
@click.option("--force", is_flag=True, default=False, help="Reprocess all PDFs (ignore checkpoint).")
@click.option(
    "--prune-missing",
    is_flag=True,
    default=False,
    help="Remove checkpoint entries for PDFs that no longer exist and delete their vectors.",
)
@click.pass_context
def ingest(ctx: click.Context, pdf_root: Path, force: bool, prune_missing: bool):
    """Incremental ingest with robust error handling."""
    s = ctx.obj["settings"]
    vs = ctx.obj["vs"]

    cp_path = checkpoint_path(s.chroma_dir)
    state = load_checkpoint(cp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pdf_paths = sorted(list(iter_pdf_paths(pdf_root)))
    if not pdf_paths:
        raise click.ClickException(f"No PDFs found under: {pdf_root}")

    # Optional: prune missing
    if prune_missing and state.get("files"):
        on_disk = {str(p) for p in pdf_paths}
        missing = [p for p in list(state["files"].keys()) if p not in on_disk]
        for missing_path in missing:
            try:
                delete_vectors_for_pdf(vs, Path(missing_path))
            except Exception as e:
                click.echo(f"⚠️  Could not delete vectors for missing file {missing_path}: {e}")
            state["files"].pop(missing_path, None)
        if missing:
            save_checkpoint(cp_path, state)
            click.echo(f"🧹 Pruned {len(missing)} missing PDFs from checkpoint.")

    processed = skipped = reprocessed = total_chunks_added = failed = 0

    with click.progressbar(pdf_paths, label="Ingesting PDFs") as bar:
        for pdf_path in bar:
            try:
                # A) classify/hash
                try:
                    if force:
                        status, h = "changed", sha256_file(pdf_path)
                    else:
                        status, h = classify_pdf(pdf_path, state)
                except Exception as e:
                    failed += 1
                    click.echo(f"\n⚠️  Failed hashing/classifying {pdf_path}: {e}")
                    continue

                if status == "unchanged":
                    skipped += 1
                    continue

                # B) delete old vectors if changed
                if status == "changed":
                    try:
                        delete_vectors_for_pdf(vs, pdf_path)
                        reprocessed += 1
                    except Exception as e:
                        failed += 1
                        click.echo(f"\n⚠️  Failed deleting old vectors for {pdf_path}: {e}")
                        continue

                # C) read + chunk
                try:
                    chunks = ingest_one_pdf(pdf_path, splitter)
                    click.echo(f"📄 {pdf_path.name}: {len(chunks)} chunks ready for embedding and chunks are {chunks}")
                except Exception as e:
                    failed += 1
                    click.echo(f"\n⚠️  Failed reading/chunking {pdf_path}: {e}")
                    continue

                if not chunks:
                    click.echo(f"\n⚠️  Skipping (no text extracted): {pdf_path}")
                    try:
                        stat = pdf_path.stat()
                        state["files"][str(pdf_path)] = {
                            "sha256": h,
                            "size": stat.st_size,
                            "mtime": stat.st_mtime,
                            "chunks": 0,
                            "last_ingested_at": now_utc_iso(),
                            "note": "no_text_extracted",
                        }
                        save_checkpoint(cp_path, state)
                    except Exception as e:
                        click.echo(f"⚠️  Failed updating checkpoint for {pdf_path}: {e}")
                    continue

                # D) embed/store
                try:
                    vs.add_documents(chunks)
                except Exception as e:
                    failed += 1
                    click.echo(f"\n⚠️  Failed embedding/storing chunks for {pdf_path}: {e}")
                    continue

                processed += 1
                total_chunks_added += len(chunks)

                # E) update checkpoint
                try:
                    stat = pdf_path.stat()
                    state["files"][str(pdf_path)] = {
                        "sha256": h,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "chunks": len(chunks),
                        "last_ingested_at": now_utc_iso(),
                    }
                    save_checkpoint(cp_path, state)
                except Exception as e:
                    click.echo(f"⚠️  Ingested but failed to update checkpoint for {pdf_path}: {e}")

            except Exception as e:
                failed += 1
                click.echo(f"\n⚠️  Unexpected failure for {pdf_path}: {e}")

    # F) persist
    try:
        vs.persist()
    except Exception as e:
        click.echo(f"⚠️  Failed to persist Chroma DB: {e}")

    click.echo(
        f"\n✅ Ingest complete\n"
        f"  PDFs found:        {len(pdf_paths)}\n"
        f"  PDFs processed:    {processed}\n"
        f"  PDFs reprocessed:  {reprocessed}\n"
        f"  PDFs skipped:      {skipped}\n"
        f"  PDFs failed:       {failed}\n"
        f"  Chunks added:      {total_chunks_added}\n"
        f"  Chroma directory:  {s.chroma_dir}\n"
        f"  Checkpoint file:   {cp_path}\n"
    )
def extract_usage(obj) -> TurnUsage:
    """
    Works across LangChain versions:
    - streaming chunks may expose usage in `usage_metadata`
    - sometimes it appears inside `response_metadata` as `token_usage` or `usage`
    """
    # 1) Newer LangChain: chunk.usage_metadata
    um = getattr(obj, "usage_metadata", None)
    if isinstance(um, dict) and um:
        return usage_from_metadata({"usage": um})

    # 2) response_metadata (varies by integration)
    rm = getattr(obj, "response_metadata", None)
    if isinstance(rm, dict) and rm:
        # usage_from_metadata already checks nested "usage" / "token_usage"
        return usage_from_metadata(rm)

    return TurnUsage()

@cli.command()
@click.option("--top-k", default=None, type=int, help="Override TOP_K from .env")
@click.option("--multiline/--singleline", default=False, help="Allow multi-line question ending with '--'.")
@click.option("--stream/--no-stream", default=True, help="Stream model output as generated.")
@click.option("--show-usage/--no-show-usage", default=True, help="Show token usage if available.")
@click.pass_context
def chat(ctx: click.Context, top_k: Optional[int], multiline: bool, stream: bool, show_usage: bool):
    summaryBasedMemory = SummaryBasedMemory(ctx.obj["llm"], "", max_cur_chat_history_size=500)
    """
    Chat:
    - Retrieves context from Chroma
    - Streams answer (no fancy UI; stable output)
    - Shows session token stats if provider returns usage
    """
    s = ctx.obj["settings"]
    vs = ctx.obj["vs"]
    llm = ctx.obj["llm"]

    k = top_k if top_k is not None else s.top_k

    try:
        retriever = vs.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        raise click.ClickException(f"Failed to create retriever: {e}")

    cur_time = now_utc_iso()
    prompt_tmpl = """
You are a helpful personal assistant for question-answering tasks against the documents shared by the user.

 - mostly Use the context below to answer and add the source of the information you used (e.g. [source=foo.pdf page=2]).
 - for question not answerable from the context, suggest that you are using your knowledge and not the context, add the source as [source=AI knowledge] and answer based on facts with source.
 - todays date added in the context answer accordingly
 
Keep the answer concise and to the point.

History: {history}

Question: {question}

Context:
---
{context}
---

Answer (1-5 sentences):
""".strip()

    stats = SessionStats()

    click.echo("RAG chat ready. Type 'exit' to quit. Type '/ml' to toggle multiline.\n")
    ml = multiline

    while True:
        q = read_question(single_line=not ml)
        if not q:
            continue

        # Commands
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "/ml":
            ml = not ml
            click.echo(f"✅ Multiline is now {'ON' if ml else 'OFF'}\n")
            continue

        print_user_bubble(q)

        # Retrieval
        try:
            hits = retrieve_docs(retriever, q)
        except Exception as e:
            click.echo(f"⚠️  Retrieval failed: {e}\n")
            continue

        # Context
        try:
            ctx_text = format_docs(hits, max_chars=s.max_context_chars)
        except Exception as e:
            click.echo(f"⚠️  Failed to format context: {e}\n")
            continue

        prompt_text = prompt_tmpl.format(question=q, context=ctx_text, history=summaryBasedMemory.get_summary())
        msg = HumanMessage(content=prompt_text)

        # Stream answer
        click.echo("Assistant:\n")
        final_usage: TurnUsage = TurnUsage()

        llm_call = llm
        final_usage = TurnUsage()
        try:
            
            if stream:
                response = ""
                for chunk in llm_call.stream([msg]):
                    delta = getattr(chunk, "content", "") or ""
                    if delta:
                        click.echo(delta, nl=False)
                        response += delta
                    u = extract_usage(chunk)
                    if u.total_tokens > 0:
                        final_usage = u  # keep the last non-zero usage (usually final chunk)

                click.echo("\n")  # newline after streaming
                summaryBasedMemory.add_to_memory(q,response)
            else:
                resp = llm_call.invoke([msg])
                click.echo((resp.content or "") + "\n")

                u = extract_usage(resp)
                if u.total_tokens > 0:
                    final_usage = u
                summaryBasedMemory.add_to_memory(q,resp.content)
            # Session stats
            if show_usage:
                if final_usage.total_tokens > 0:
                    stats.update(final_usage)
                    click.echo(stats.format_summary(final_usage) + "\n")
                else:
                    click.echo("📌 Token usage: not available (streaming usage not surfaced by this LangChain/OpenAI version).\n")
                    
        except Exception as e:
            click.echo(f"\n⚠️  LLM call failed: {e}\n")
            continue


if __name__ == "__main__":
    cli()