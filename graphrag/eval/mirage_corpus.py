"""Prepare and query the official MedRAG Textbooks corpus locally.

The 18.9 GB MIRAGE top-10k archive contains precomputed IDs for many
corpus/retriever combinations. For a laptop-sized, reproducible matched-corpus
evaluation, this module uses the official 18-book Textbooks corpus (about
202 MiB) and builds a dependency-free SQLite FTS5 BM25 index.

Corpus files and the generated index stay under the gitignored external-data
directory. Only the pinned source manifest, adapter, and tests are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()

DEFAULT_MANIFEST = Path(__file__).parent / "specs" / "medrag_textbooks_manifest.json"
_configured_external_root = os.getenv("MIRAGE_EXTERNAL_ROOT", "").strip()
_external_root = (
    Path(_configured_external_root).expanduser()
    if _configured_external_root
    else Path(__file__).parent / "external" / "mirage"
)
DEFAULT_ROOT = _external_root / "textbooks"
DEFAULT_CORPUS_DIR = DEFAULT_ROOT / "corpus"
DEFAULT_INDEX = DEFAULT_ROOT / "textbooks_bm25.sqlite3"
INDEX_FORMAT_VERSION = 1

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could",
    "does", "for", "from", "has", "have", "how", "in", "is", "it", "may",
    "most", "not", "of", "on", "or", "patient", "should", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "which", "with",
    "would",
}


@dataclass(frozen=True)
class TextbookSnippet:
    snippet_id: str
    title: str
    content: str
    score: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"format_version", "name", "revision", "source", "files"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError("Invalid MedRAG Textbooks manifest")
    if manifest["format_version"] != 1 or not isinstance(manifest["files"], list):
        raise ValueError("Unsupported MedRAG Textbooks manifest version")
    seen: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError("Invalid file entry in MedRAG Textbooks manifest")
        if (
            not str(item["path"]).startswith("chunk/")
            or item["path"] in seen
            or int(item["size"]) <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"]))
        ):
            raise ValueError(f"Invalid manifest file: {item!r}")
        seen.add(item["path"])
    return manifest


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def verify_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    manifest: dict[str, Any] | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    manifest = manifest or load_manifest()
    valid: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    for item in manifest["files"]:
        path = corpus_dir / item["path"]
        if not path.exists():
            missing.append(item["path"])
            continue
        if path.stat().st_size != item["size"]:
            invalid.append(item["path"])
            continue
        if verify_hashes and _sha256(path) != item["sha256"]:
            invalid.append(item["path"])
            continue
        valid.append(item["path"])
    return {
        "complete": not missing and not invalid,
        "valid_files": len(valid),
        "total_files": len(manifest["files"]),
        "expected_bytes": sum(item["size"] for item in manifest["files"]),
        "missing": missing,
        "invalid": invalid,
        "manifest_fingerprint": manifest_fingerprint(manifest),
    }


def download_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or load_manifest()
    base = "https://huggingface.co/datasets/MedRAG/textbooks/resolve"
    for position, item in enumerate(manifest["files"], start=1):
        destination = corpus_dir / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.exists()
            and destination.stat().st_size == item["size"]
            and _sha256(destination) == item["sha256"]
        ):
            print(f"[{position}/{len(manifest['files'])}] {item['path']}: cached")
            continue
        if destination.exists():
            raise ValueError(
                f"Existing corpus file failed integrity check: {destination}. "
                "Move it aside before retrying."
            )

        url = (
            f"{base}/{manifest['revision']}/{quote(item['path'])}?download=true"
        )
        temporary = destination.with_suffix(destination.suffix + ".download")
        request = Request(url, headers={"User-Agent": "medical-graphrag-eval/1.0"})
        print(
            f"[{position}/{len(manifest['files'])}] {item['path']}: "
            f"downloading {item['size'] / 1024 / 1024:.1f} MiB"
        )
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if temporary.stat().st_size != item["size"]:
                raise ValueError(f"Size mismatch for {item['path']}")
            if _sha256(temporary) != item["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {item['path']}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return verify_corpus(corpus_dir, manifest)


def _require_complete_corpus(
    corpus_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    verification = verify_corpus(corpus_dir, manifest)
    if not verification["complete"]:
        raise ValueError(
            "MedRAG Textbooks corpus is incomplete; run mirage-corpus --download first"
        )
    return verification


def build_bm25_index(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    index_path: Path = DEFAULT_INDEX,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or load_manifest()
    verification = _require_complete_corpus(corpus_dir, manifest)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".building")
    temporary.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary)
    snippet_count = 0
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE snippet_ids (snippet_id TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE VIRTUAL TABLE snippets USING fts5(
                snippet_id UNINDEXED,
                title,
                content,
                tokenize='porter unicode61 remove_diacritics 2'
            );
            """
        )
        for position, item in enumerate(manifest["files"], start=1):
            rows: list[tuple[str, str, str]] = []
            with (corpus_dir / item["path"]).open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON at {item['path']}:{line_number}"
                        ) from exc
                    snippet_id = str(row.get("id") or "").strip()
                    title = str(row.get("title") or "").strip()
                    content = str(row.get("content") or "").strip()
                    if not snippet_id or not title or not content:
                        raise ValueError(
                            f"Missing snippet fields at {item['path']}:{line_number}"
                        )
                    rows.append((snippet_id, title, content))
                    if len(rows) >= 1000:
                        _insert_rows(connection, rows)
                        snippet_count += len(rows)
                        rows.clear()
                if rows:
                    _insert_rows(connection, rows)
                    snippet_count += len(rows)
            connection.commit()
            print(
                f"[{position}/{len(manifest['files'])}] indexed {item['path']} "
                f"({snippet_count:,} snippets total)"
            )
        connection.execute("INSERT INTO snippets(snippets) VALUES('optimize')")
        metadata = {
            "index_format_version": str(INDEX_FORMAT_VERSION),
            "corpus_name": str(manifest["name"]),
            "corpus_revision": str(manifest["revision"]),
            "manifest_fingerprint": verification["manifest_fingerprint"],
            "snippet_count": str(snippet_count),
            "retriever": "sqlite-fts5-bm25",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
        os.replace(temporary, index_path)
    return index_metadata(index_path)


def _insert_rows(connection: sqlite3.Connection, rows: list[tuple[str, str, str]]) -> None:
    connection.executemany(
        "INSERT INTO snippet_ids(snippet_id) VALUES (?)",
        [(snippet_id,) for snippet_id, _, _ in rows],
    )
    connection.executemany(
        "INSERT INTO snippets(snippet_id, title, content) VALUES (?, ?, ?)", rows
    )


def index_metadata(index_path: Path = DEFAULT_INDEX) -> dict[str, Any]:
    if not index_path.exists():
        return {"exists": False, "path": str(index_path)}
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    finally:
        connection.close()
    return {
        "exists": True,
        "path": str(index_path),
        "size_bytes": index_path.stat().st_size,
        **metadata,
    }


def _fts_query(text: str) -> str:
    tokens = re.findall(r"[a-z][a-z0-9]+|\d+(?:\.\d+)?", text.lower())
    unique: list[str] = []
    for token in tokens:
        if token in _STOPWORDS or token in unique:
            continue
        unique.append(token)
        if len(unique) >= 40:
            break
    if not unique:
        raise ValueError("Retrieval query has no searchable terms")
    return " OR ".join(f'"{token}"' for token in unique)


class TextbooksBM25Retriever:
    def __init__(self, index_path: Path = DEFAULT_INDEX):
        metadata = index_metadata(index_path)
        if not metadata.get("exists"):
            raise FileNotFoundError(
                f"Textbooks BM25 index not found: {index_path}. "
                "Run mirage-corpus --download --build-index."
            )
        if int(metadata.get("index_format_version", -1)) != INDEX_FORMAT_VERSION:
            raise ValueError("Unsupported Textbooks BM25 index version")
        self.index_path = index_path
        self.metadata = metadata

    def retrieve(self, query: str, k: int = 8) -> list[TextbookSnippet]:
        if k <= 0:
            raise ValueError("k must be positive")
        connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """
                SELECT snippet_id, title, content,
                       bm25(snippets, 0.0, 0.2, 1.0) AS rank
                FROM snippets
                WHERE snippets MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (_fts_query(query), k),
            ).fetchall()
        finally:
            connection.close()
        return [
            TextbookSnippet(
                snippet_id=row[0], title=row[1], content=row[2], score=-float(row[3])
            )
            for row in rows
        ]


def _summary(
    corpus_dir: Path, index_path: Path, manifest: dict[str, Any], verify_hashes: bool
) -> dict[str, Any]:
    return {
        "corpus": manifest["name"],
        "revision": manifest["revision"],
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "source_files": len(manifest["files"]),
        "expected_bytes": sum(item["size"] for item in manifest["files"]),
        "download": verify_corpus(corpus_dir, manifest, verify_hashes=verify_hashes),
        "index": index_metadata(index_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if args.download:
        download_corpus(args.corpus_dir, manifest)
    if args.build_index:
        build_bm25_index(args.corpus_dir, args.index, manifest)
    if args.query:
        snippets = TextbooksBM25Retriever(args.index).retrieve(args.query, args.top_k)
        for position, snippet in enumerate(snippets, start=1):
            print(
                f"[{position}] {snippet.snippet_id} score={snippet.score:.4f}\n"
                f"{snippet.content[:500]}\n"
            )
    if args.dry_run or not (args.download or args.build_index or args.query):
        print(
            json.dumps(
                _summary(args.corpus_dir, args.index, manifest, verify_hashes=False),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
