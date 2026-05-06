"""
FAISS vector retriever
----------------------
Builds a FAISS flat index over guidelines.txt chunks.
Uses the same all-mpnet-base-v2 model as the baseline for continuity.

Usage:
    vr = VectorRetriever()
    results = vr.retrieve("patient has fever and cough", k=5)
    # returns: list of (score: float, text: str)
"""

import os
import pickle
import re
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from graphrag.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, GUIDELINES_PATH, TOP_K_VECTOR


@dataclass
class VectorResult:
    score: float
    text: str


class VectorRetriever:
    def __init__(self, force_rebuild: bool = False):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []

        index_file = FAISS_INDEX_PATH + ".faiss"
        chunks_file = FAISS_INDEX_PATH + ".pkl"

        if not force_rebuild and os.path.exists(index_file) and os.path.exists(chunks_file):
            self._load(index_file, chunks_file)
        else:
            self._build_and_save(index_file, chunks_file)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = TOP_K_VECTOR) -> list[VectorResult]:
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(VectorResult(score=float(score), text=self.chunks[idx]))
        return results

    # ------------------------------------------------------------------
    def _chunk_guidelines(self) -> list[str]:
        with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = re.split(r"\n{2,}", text.strip())
        return [p.strip() for p in paragraphs if len(p.strip()) >= 80]

    def _build_and_save(self, index_file: str, chunks_file: str) -> None:
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        print("[vector] Building FAISS index...")
        self.chunks = self._chunk_guidelines()
        embeddings = self.model.encode(self.chunks, normalize_embeddings=True, show_progress_bar=True)
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine sim on normalized vecs
        self.index.add(embeddings)
        faiss.write_index(self.index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[vector] Index built: {len(self.chunks)} chunks, dim={dim}")

    def _load(self, index_file: str, chunks_file: str) -> None:
        self.index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[vector] Loaded FAISS index: {len(self.chunks)} chunks")
