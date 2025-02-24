import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


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

def get_answer_from_ollama(query: str, context_chunks: list[str]) -> str:
    """
    Get answer from Ollama model using the query and context chunks
    
    Args:
        query: The user's question
        context_chunks: List of relevant text chunks for context
        
    Returns:
        Answer from the Ollama model
    """

    # Join context chunks into single string
    context = "\n\n".join(context_chunks)
    
    # Initialize Ollama model
    llm = Ollama(model="llama3")
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use the following context to answer the question. 
                      If you cannot find the answer in the context, say exactly "I dont know" and nothing else.
                      \n\nContext:\n{context}"""),
        ("user", "Question: {question}")
    ])
    print("\n\n\n\n",context,"\n\n\n\n")
    # Create chain and get response
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })
    
    return response


if __name__ == "__main__":

    print("Hello. Ask me anything")
    vectordb_path = "VectorDb/faiss_index.db"

    while True:
        print("You :", end = " ")
        userquestion = input()
        if userquestion == "bye":
            break
        similar_chunks = search_similar_chunks(userquestion, vectordb_path)
        answer = get_answer_from_ollama(userquestion, similar_chunks)
        print(answer)   