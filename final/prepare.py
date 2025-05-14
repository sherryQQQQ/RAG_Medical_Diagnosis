from txtai.embeddings import Embeddings
import os
from typing import List, Tuple

class DocumentVectorizer:
    def __init__(self):
        self.document = open('guidelines.txt', 'r').read()  # TODO: store the guidelines.txt as a string to this variable

    def chunk_text(self) -> List[str]:
        '''
        The text input is the string representation of the guidelines. 
        
        TODO: implement a text chunking method
        This function should return a list of strings, where each string is a chunk of the original text.
        It is up to you to determine how to chunk the text. 
        Your chunking method will affect the quality of the embeddings, and therefore the quality of the retrieval.
        '''
        
        text = self.document
        # chunks = ['replace this list with your chunks and return it']
        # chunks = [line.strip() for line in text.split('\n\n') if line.strip()]
        # chunks2 = []
        # for i in range(len(chunks)):
        #     temp = chunks[i].split('.')
        #     for j in range(len(temp)-1):
        #         chunk2 = temp[j]+temp[j+1]
        #         chunks2.append(chunk2)
        chunks = [s.strip() for s in text.split('\. ') if s.strip()]
        # chunks = [line.strip() for line in text.split('.') if line.strip()]
        return chunks
    
    def generate_embeddings(self, text_chunks: List[str]) -> Embeddings:
        """Create a txtai embeddings database and save it to disk.
        
        Each chunk is a string, and this function embeds each chunk. It should 
        return an Embeddings object. 
        
        TODO: chose the model that will be used to embed the text chunks.
        """
        os.makedirs('db', exist_ok=True)
        
        # what model should we use?
        embeddings = Embeddings(path="sentence-transformers/all-mpnet-base-v2") # TODO: we should call the Embeddings class here
        # an index step is needed here too
        # embeddings=Embeddings(path="sentence-transformers/all-mpnet-base-v2")
        # data=[(i,chunk) for i,chunk in enumerate(text_chunks)]
        embeddings.index(text_chunks)
        embeddings.save(f"db/embeddings")
        
        return embeddings

    def load_database(self) -> Tuple[Embeddings, List[str]]:
        """Load the embeddings database and articles from disk.
        
        This function returns the embeddings and the corresponding text chunks.
        It loads the embeddings from your hard drive which is originally saved in the 
        generate_embeddings function.
        
        TODO: pick the model that will be used to load the embeddings. This should be 
        the same model that you used to generate the embeddings.
        """
        
        # What model should we use?
        embeddings = Embeddings(path="sentence-transformers/all-mpnet-base-v2") #  we should call the Embeddings class here
        embeddings.load(f"db/embeddings")
        text_chunks = self.chunk_text()
        
        return embeddings, text_chunks