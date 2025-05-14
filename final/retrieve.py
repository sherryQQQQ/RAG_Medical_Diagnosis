from typing import List, Tuple
from prepare import DocumentVectorizer

class DocumentRetriever:
    def __init__(self, limit=2):
        self.dv = DocumentVectorizer()
        self.limit = limit
        # TODO: Finish the __init__ function
        # you need to generate embeddings here 
        self.text_chunks = self.dv.chunk_text()
        self.embeddings = self.dv.generate_embeddings(self.text_chunks)
        # self.embeddings,self.text_chunks = self.dv.load_database()

    def retrieve(self, query) -> List[Tuple[float, str]]:
        '''
        This function retrieves the most relevant documents to the query from the guidelines.txt file.
        It uses the embeddings to search for the most relevant documents.
        
        It retrieves the top (self.limit) documents from the guidelines.txt file. The default limit is 1, 
        so it will return the most relevant document to the query (with respect to the embedding space.)
        
        TODO: implement this function as described above.
        
        Your return must be a list of tuples, where each tuple contains the similarity score and the text chunk associated 
        with the score.
        
        Hint: you already built the document vectorizer in the prepare.py file. 
        It is called for you in the __init__() function of this class. 
        What can you do with the document vectorizer to get the most relevant documents?
        
        Use the search capability of the Embeddings library from txtai to retrieve the most relevant documents.
        '''

        embeddings,text_chunks = self.dv.load_database() # TODO: store the guidelines.txt as a string to this variable
        results =  embeddings.search(query,limit=self.limit) 
        return [(score, text_chunks[uid]) for uid, score in results]