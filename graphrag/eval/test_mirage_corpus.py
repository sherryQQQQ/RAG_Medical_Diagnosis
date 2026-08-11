import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from graphrag.eval.mirage_corpus import (
    TextbooksBM25Retriever,
    build_bm25_index,
    load_manifest,
    verify_corpus,
)


class MirageCorpusTests(unittest.TestCase):
    def _fixture(self, directory: str):
        root = Path(directory)
        corpus = root / "corpus"
        source = corpus / "chunk" / "fixture.jsonl"
        source.parent.mkdir(parents=True)
        rows = [
            {
                "id": "fixture_0",
                "title": "Gastroenterology",
                "content": "Barrett esophagus is associated with chronic acid reflux.",
            },
            {
                "id": "fixture_1",
                "title": "Cardiology",
                "content": "Cardiac myocytes undergo hypertrophy in hypertension.",
            },
            {
                "id": "fixture_2",
                "title": "Neurology",
                "content": "Myelin supports saltatory nerve conduction.",
            },
        ]
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        source.write_text(payload, encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "name": "fixture/textbooks",
                    "revision": "test-revision",
                    "source": "https://example.test",
                    "files": [
                        {
                            "path": "chunk/fixture.jsonl",
                            "size": source.stat().st_size,
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return corpus, manifest_path, root / "index.sqlite3"

    def test_build_and_query_bm25_index(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus, manifest_path, index = self._fixture(directory)
            manifest = load_manifest(manifest_path)
            metadata = build_bm25_index(corpus, index, manifest)
            results = TextbooksBM25Retriever(index).retrieve(
                "Does chronic acid reflux cause Barrett esophagus?", k=2
            )

        self.assertEqual(metadata["snippet_count"], "3")
        self.assertEqual(results[0].snippet_id, "fixture_0")
        self.assertGreater(results[0].score, 0)

    def test_verify_detects_missing_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus, manifest_path, _ = self._fixture(directory)
            manifest = load_manifest(manifest_path)
            self.assertTrue(verify_corpus(corpus, manifest)["complete"])
            source = corpus / "chunk" / "fixture.jsonl"
            source.write_text("x" * source.stat().st_size, encoding="utf-8")
            result = verify_corpus(corpus, manifest)

        self.assertFalse(result["complete"])
        self.assertEqual(result["invalid"], ["chunk/fixture.jsonl"])

    def test_manifest_rejects_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest_path, _ = self._fixture(directory)
            payload = json.loads(manifest_path.read_text())
            payload["files"].append(dict(payload["files"][0]))
            manifest_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Invalid manifest file"):
                load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
