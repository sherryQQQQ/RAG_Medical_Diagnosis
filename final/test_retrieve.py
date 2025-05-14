import unittest
import os

from txtai.embeddings import Embeddings

from prepare import DocumentVectorizer
from retrieve import DocumentRetriever
from generate import Gemini, Generator

class TestCases(unittest.TestCase):
    
    def test_retriever(self):
        dr = DocumentRetriever()
        response = dr.retrieve("What is the meaning of life?")
        expected_response = [(score, text) for score, text in response]
        self.assertIsInstance(expected_response, list)
        self.assertIsInstance(expected_response[0], tuple)
        self.assertEqual(len(expected_response[0]), 2)  # Each tuple should have two elements: score and text
