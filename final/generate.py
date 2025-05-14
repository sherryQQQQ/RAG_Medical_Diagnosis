import google.generativeai as genai
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_random
import json

from retrieve import DocumentRetriever

class Gemini:
    def __init__(self, model_name: str):
        self.model = model_name
        self.api_key = '########'
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self.model)
        
    async def query_async(self, prompt):
        response = await self.client.generate_content_async(prompt)
        return response.text
    
    def query(self, prompt):
        '''
        This function is a wrapper around the Gemini API.
        It takes a string prompt and returns a response from Gemini.
        
        
        Your response must be a string. 
        If unclear what to do, refer to the function line 17 as a hint.
        '''
        response = self.client.generate_content(prompt) # TODO: query Gemini to generate a response to the prompt
        return response.text



class Generator:
    def __init__(self):
        self.model = Gemini("gemini-1.5-flash-latest")
        self.dr = DocumentRetriever()

    def process_query(self, query, documents):
        '''
        This function is takes a list of documents and a query, and returns a response from Gemini.
        
        write a prompt that will generate the answer to the question given the documents 
        Keep the lines begining with "Your response should be a json object in the following format:"
        until the sentence that says it is imperative that you only return the json object in the following format.
        
        After writing the prompt, you may choose to optimize the prompt with query rewriting.
        
        Remember to format your prompt with string formatting.
        
        Replace where it says "BEGIN PROMPT HERE" with your prompt.
        '''
        
        prompt = f"""
            As a medical expert, your task is to provide a diagnosis or recommendation based solely on the provided guidelines. Do not use any external knowledge.

            GUIDELINES:
            {documents}

            PATIENT QUERY:
            {query}

            Follow these steps to generate your response:
            1. Understand the patient’s query clearly.
            2. Identify key symptoms, conditions, or medical terms in the query.
            3. Find exact quotes in the guidelines and write them word for word, ensuring the conditions match precisely.
            4. If you cannot find the information, search the guidelines again since there should be a match.
            5. If you still think the information is missing, classify the query as "miscellaneous yet evolving topics" and look for relevant guideline details.
            6. Internally consider possible diagnoses based on the guidelines.
            7. Document internal reasoning leading to the final diagnosis based on the guidelines(this will not be shown to the user).
            8.  Before finalizing the answer to the query, conduct a strict accuracy verification:
                - Ensure all numbers, durations, medication dosages, and procedural steps match the guidelines exactly.
                - Cross-check whether additional steps (e.g., follow-up tests, contraindications) are necessary.
                - Ensure all test orders, treatment durations, and procedural steps are listed completely and correctly.
                - If a multi-step process is involved, present it step by step.
            9. Include all necessary medical steps (e.g., tests, treatments,numbers).
                - Do not introduce extra conditions, tests, or treatments unless stated in the query.
            10. Structure your response in a clear, concise manner:
                - State the final diagnosis or recommendation in a direct and authoritative sentence as an answer to the query.
                - If needed, include the exact treatment plan (medication, duration, or steps).
                - If a test is required, specify the exact test name.
                - If follow-up is necessary, specify when and what to check.
                - Do not add extra information or background unless explicitly asked.

            Your response must be a JSON object in the following format:
            {{
            "answer": "<diagnosis or recommendation>"
            }}
            It is imperative that you only return the JSON object in the specified format, or you will be penalized.
            """
        # prompt = None # TODO: implement the string formatting necessary to produce your prompt
        
        # Nothing else needs to be done beyong this line for this function.
        prompt = self.optimize_query(prompt) # This is currently returning the original prompt. Optionally, you can choose to optimize the prompt.
        response = self.model.query(prompt) 
        response = response[response.find("{"):response.find("}")+1]
        response = json.loads(response)
        return response['answer']
    
    def optimize_query(self, original_query):
        '''
        Query optimization is optional, you may not need to implement this. 
        However, it may boost your RAG performance. Keep in mind this will
        increrase the time it takes to generate a response as it requires
        making another API call. 
        
        This function should take in the original query and return a new query
        that has been somehow improved. 
        
        You are free to use the Gemini API to have gemini optimize the query,
        but you will need to build the pipeline yourself. 
        
        For reference on how to implement this, please see the process_query function.
        '''        
        # TODO: (OPTIONAL) implement this function
        optimized_query = self.model.query(original_query)
        validation_prompt=f"""            
        - Answering the specific clinical question directly—avoid just explaining background concepts.
            - Double-checking for missing details—like treatment durations, stepwise processes, or test orders.
            - Ensuring accuracy in key recommendations—even small details (e.g., test order, medication discontinuation timing) can change the clinical impact.
            - Avoiding unnecessary background explanations.
        """
        validation_response = self.model.query(validation_prompt)
        if validation_response == 'VALID':
            return optimized_query
        else:
            return original_query

    