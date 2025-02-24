import faiss
import pickle
import openai
import os
import numpy as np
from openai import AzureOpenAI
from azure.identity import EnvironmentCredential

from sentence_transformers import SentenceTransformer




def search_similar_chunks(query: str, index_path: str, k: int = 3) -> list[str]:
    """
    Search for chunks similar to the query in the FAISS index
    
    Args:
        query: The search query string
        index_path: Path to the FAISS index file
        k: Number of similar chunks to return (default 3)
    
    Returns:
        List of similar text chunks
    """
    # Load the FAISS index
    index = faiss.read_index(index_path)

    # Load the metadata containing text chunks
    with open(index_path + '.metadata', 'rb') as f:
        metadata = pickle.load(f)
    chunks = metadata['chunks']

    # Initialize the model and generate embedding for query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    # Convert to correct format and type
    query_embedding = np.array(query_embedding).astype('float32')

    # Search the index
    distances, indices = index.search(query_embedding, k)

    # Get the corresponding text chunks
    similar_chunks = [chunks[idx] for idx in indices[0]]

    return similar_chunks

def __get_access_token():
    credential = EnvironmentCredential()
    access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return access_token.token



def get_answer_from_openai(query: str, context_chunks: list[str]) -> str:
    """
    Get answer from OpenAI model using the query and context chunks
    
    Args:
        query: The user's question
        context_chunks: List of relevant text chunks for context
        
    Returns:
        Answer from the OpenAI model
    """

    response = None
    # Join context chunks into single string
    context = "\n\n".join(context_chunks)



    strPrompt = """You are a helpful library assistant who can summarize answers from various pieces of text. Your inability to answer a question is not a concern 
                    but wrong answer carries a heavy penalty. So, it is important to not assume any information. If you do not have enough information, say 
                    "I cannot answer the question based on the information provided.". Read the below text chunks and answer the question " """ + query + """ " """ + "\n\nContext:\n" + context + """ """

    os.environ['AZURE_OPENAI_ENDPOINT'] = "https://cog-sandbox-dev-eastus2-001.openai.azure.com/"
    client = AzureOpenAI(api_version="2023-05-15", api_key=__get_access_token())

    messages= [{'role':'user', 'content':strPrompt}]
    response = client.chat.completions.create(model="gpt-35-turbo-blue", messages=messages)

    print(strPrompt)


    # end of my own code
    return response.choices[0].message.content


if __name__ == "__main__":

    print("Hello. Ask me anything")
    vectordb_path = "VectorDb/faiss_index.db"

    while True:
        print("You :", end = " ")
        userquestion = input()
        if userquestion == "bye":
            break
        similar_chunks = search_similar_chunks(userquestion, vectordb_path)
        answer = get_answer_from_openai(userquestion, similar_chunks)
        print(answer)   