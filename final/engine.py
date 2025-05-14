import pandas as pd
import json

from generate import Generator
from retrieve import DocumentRetriever

if __name__ == "__main__":
    question_df = pd.read_csv("medical_generalization.csv", index_col=0)
    question_df = question_df.iloc[70:80,:] # just check the first 10 questions to save time.  you may change this number to check more/less questions.
    
    dr = DocumentRetriever()
    g = Generator()
    responses = []
    documents = []
    for i, row in question_df.iterrows():
        question = row["prompt"]
        docs = dr.retrieve(question)
        documents.append(docs)
        
        response = g.process_query(question, docs)
        responses.append(response)
    list_of_responses = []
    for response, d, question, answer in zip(responses, documents, question_df['prompt'], question_df['answer']):
        print("Question: ", question)
        print("Your Answer: ", response)
        print("Correct Answer: ", answer)
        print("Documents: ", d)
        print("--------------------------------")
        list_of_responses.append(
                'question:'+ str(question)+'\n\n'+
                'your_answer:'+ str(response)+'\n'+
                'correct_answer:'+ str(answer)+'\n'+
                'documents'+ str(d)+'\n\n'
        )
    # save to txt
    with open('list_of_responses.txt', 'w') as f:
        for response in list_of_responses:
            f.write(response)
            f.write('\n')
        