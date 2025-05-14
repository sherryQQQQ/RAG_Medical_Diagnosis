Please read the entire README before beginning the project.

## Final Project 

Your job is to produce medical diagnoses for patient queries. You will be given a set of patient casefiles briefs, and a document of medical guidelines. This dataset is in the `medical_generalization.csv` file.  The medical question (what we refer to as casefile) is in the `prompt` column. The correct answer is in the `answer` column.

For a given patient query `prompt`, you will need to implement a RAG pipeline to generate a diagnosis/recommendation. 

Unlike the previous homeworks where a lot of the code was provided for you, this project requires a little more involvement.  

You will have functions to implement in the following files:

- `prepare.py`: 
    - the `__init__()` function should read the guidelines.txt file and store it as a string in the `document` variable. 
    - the `chunk_text()` function should take the `document` variable and return a list of strings; chunked document. 
    - the `generate_embeddings()` function should take the list of strings and return an Embeddings object. You will need to implement the Embeddings
    - the `load_database()` function should load the embeddings from disk and return the embeddings and the corresponding text chunks.

- `retrieve.py`: You will need to implement the `retrieve` function. This function will retrieve the most relevant documents to the query. 
    - the `__init__()` needs to build the vector database by calling functions from the class `DocumentVectorizer` class in `prepare.py` file.
    - the `retrieve()` function needs to take a query and return the most relevant documents to the query. 

- `generate.py`: You will need to implement the `query` function. This function will generate a response to the query.
    - the `__init__()` function needs to set up the Gemini API.  You will need to use your Gemini API key here.
    - the `query()` function needs to take a query and pass it to the Gemini API to return a response.
    - the `process_query()` function needs a prompt, which you will write. It will then be passed to the Gemini API to return a response.
    - the `optimize_query()` function is optional, but you can implement it if you want to optimize the query. 
    
We provide a few unit tests to help you ensure your implemented code is at least returning the correct output. 
You can run these tests by running `python -m run_tests.py` in the terminal.


### Prepare Vector Database `prepare.py`

Load in the `guidelines.txt` file as a string in the `self.document` variable.

The first step is to prepare the vector database. This will allow you to retrieve the most relevant documents to the query.

#### Text preprocessing

You have a large document of medical guidelines. You cannot directly use this document to generate embeddings, because it is too large
and embedding the entire document as one vector will lead to a lot of information loss. 

Instead, you will need to preprocess the document into smaller chunks. Complete the `chunk_text` function in the `prepare.py` file. This function should take the entire document as input, and return a list of strings. For this project, it is enough to use native python libraries to perform the chunking. Do not use any external libraries for chunking, as we will not be able to run your code in the grading environment, which will ultimately result in a score of 0.

#### Generate Embeddings

Once you have preprocessed the document into smaller chunks, you will need to generate the embeddings for the document. 

Complete the `generate_embeddings` function in the `prepare.py` file. This function should take the list of strings as input, and return an Embeddings object. 

Your job is to pick the right model for the embeddings. You should experiment with different models after the whole pipeline is implemented, and 
see which model yields the best results. 

Refer to the [sentence-transformers](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) for more information on the available models.

You will also need to use the same model to load the embeddings from disk in the `load_database` function.

#### Load Embeddings

You will need to load the embeddings from disk in the `load_database` function in later steps. This function should take the embeddings from disk and return the embeddings and the corresponding text chunks.

### Retrieve Documents `retrieve.py`

You will need to complete the `__init__` function in the `retrieve.py` file. This should set up your vector database. Additionally, it takes in a `limit` keyword argument. The default value for `limit` is 1, so it will return the most relevant document to the query. However, you can change this value.

There is an additinal function in the `retrieve.py` file, `retrieve`. This function will take a query and return the most relevant documents to the query. 

This function takes a query and a `limit` keyword argument. The default value for `limit` is 1, so it will return the most relevant document to the query. You should allow the retrieval step to follow the `self.limit` value.

Your output should be a list of tuples, where each tuple contains the similarity score and the text chunk associated with the score.


### Generate Response `generate.py`

There are two classes in the `generate.py` file, `Gemini` and and `Generator`

The `Generator` class is the main class for this homework. It will take a query and return a response. 

The `Gemini` class is a wrapper around the Gemini API. It will take a prompt and return a response.

For `Gemini`, you will need to set your api key as you have been doing in previous assignments. You are also responsible for setting up the query function.

In the `Generator` class, you will need to set up the `process_query` function. This function will take a query and return a response. You can also optionally set up the `optimize_query` function. This function will take an original prompt and return an optimized prompt. Currently, it is taking in the original prompt, and returning the original prompt. 

#### Note:

In `process_query`, you will need to set up the prompt. The prompt looks like this. 

It is important keep the lines from `Your response should be a json object in the following format:` to `It is imperative that you only return the json object in the following format...` in your prompt. This is how we will extract the answer from the LLM response. Your job is to replace the `BEGIN PROMPT HERE` with your prompt.

```
BEGIN PROMPT HERE


Your response should be a json object in the following format:
{{
    "answer": <answer>,
}}

It is imperative that you only return the json object in the following format
or else you will be penalized.
```

### Evaluate Response

You can run the `engine.py` file to evaluate your response. This takes sample questions and compares your response to the correct answer. Your sample questions come from the `medical_generalization.csv` file.

The file can be run by running `python -m engine` in your working directory.

Note that the `engine.py` file is not graded, and you should not submit this file. Running this script will give you an idea of how well your RAG is performing. During grading, 
we will use a different set of questions to evaluate your response. Using such a holdout set is standard practice in machine learning in general. You will not be able to see the questions or answers that are used for grading.


### Grading

First, you will be scored on the quality of your response, i.e. are you returning the correct answer.

Second, you will be scored on the relevance of your documents. Each question in the test set is associated with certain facts from the guidelines. If the relevant guideline is your n-th retrieved document, you will receive 1/n points. That is, if the relevant chunk of text is in your first retrieved chunk, you will receive 1 point (the maximum score for the given question). If it is in the second place, you will receive 0.5 points, etc.... We will average this relevancy score across all questions.

This is very similar to the [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank). 

Your score will be the average of the quality score and the relevancy score. Do not be alarmed that your score seems lower than expected. This is a noisy dataset and reflects the problems that are faced by real world RAG systems. Your score will be curved, but you should aim to get as close to 1.0 as possible. 
 

### Submission

You are to submit the following files to the autograder:

- `prepare.py`
- `retrieve.py`
- `generate.py`

Any other files you submit will be ignored.

**You are only allowed to submit to Gradescope a maximum of 3 times. Submissions after the 3rd submission will be ignored and given a score of 0. Gradescope errors will not be counted against you, but you are responsible for your score on the Gradescope test cases.**


### Testing

You can test your code by running `python -m run_tests` in the terminal. These test are built for your convenience, but they are not exhaustive, nor do they check if your RAG system is good. It simply checks that your return types are correct, so that your code is compatible with the `engine.py` file. There is no guarantee that passing these tests mean your implementation is correct. You are free to add more of your own tests to the `run_tests.py` file provided it follows the `unittest` framework.

If you get the following output when runnning the tests

```
......
----------------------------------------------------------------------
Ran 6 tests in 10.365s

OK
```

This means that all the tests passed. If you get any other output, you should follow the error messages to fix your code.


### Note on the project:

A lot of the code you are to write for this project exists in your previous assignments. Please refer to the previous assignments for further hints. 

As such, we will not be providing you with further hints or answering questions about how to solve this project. If you run into issues with bugs, i.e. low level errors, technical issues with gradescope, we are available to answer these question. However, we will not be answering questions on debugging your code, or how to implement a certain feature. 

You have learned a lot of concepts in this course and have done very well. We are confident that you are able to leverage what you have learned to complete this project.
