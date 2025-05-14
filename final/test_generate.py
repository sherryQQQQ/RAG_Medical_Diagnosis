import unittest
import json

from txtai.embeddings import Embeddings

from generate import Gemini, Generator

class TestCases(unittest.TestCase):
    
    def test_gemini(self):
        g = Gemini(model_name="gemini-1.5-flash-002")
        response = g.query("What is the meaning of life?")
        self.assertIsInstance(response, str)
        
    def test_generator(self):
        g = Generator()
        documents = [
            "The meaning of life is subjective and can vary from person to person.",
            "The meaning of life is centered around concepts of serving a higher power, achieving enlightenment, or preparing for an afterlife.",
            "The meaning of life is to survive, reproduce, and pass on genes.",
            "The meaning of life is achieving personal fulfillment, relationships, love, creativity, and contributing to society.",
            "The meaning of life is a deeply personal question that each individual must answer for themselves.",
        ]
        response = g.process_query("What is the meaning of life?", documents)

        
        self.assertIsInstance(response, str) 

        
        
        