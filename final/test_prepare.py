import unittest
import os

from txtai.embeddings import Embeddings

from prepare import DocumentVectorizer

class TestCases(unittest.TestCase):
    
    def test_chunker(self):
        # chunk should return a list of strings 
        # this test only checks that the chunker returns a list of strings
        # what ultimately matters is the chunker should work well with the RAG system 
        dv = DocumentVectorizer()
        chunked = dv.chunk_text()
        self.assertIsInstance(chunked, list)
        self.assertIsInstance(chunked[0], str)
    
    def test_database_generator(self):
        dv = DocumentVectorizer()
        embeddings = dv.generate_embeddings(dv.chunk_text())
        self.assertTrue(os.path.exists("db/embeddings"), "The 'db' directory does not exist.")
        self.assertIsInstance(embeddings, Embeddings)
    
    def test_database_loader(self):
        dv = DocumentVectorizer()
        embeddings, text_chunks = dv.load_database()
        self.assertIsInstance(embeddings, Embeddings)
        self.assertIsInstance(text_chunks, list)
        self.assertIsInstance(text_chunks[0], str)
        