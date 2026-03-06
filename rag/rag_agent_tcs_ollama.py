# import autogen
# import faiss
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer

# def search_similar_chunks(query: str, index_path: str, k: int = 3) -> list[str]:
#     """
#     Search for chunks similar to the query in the FAISS index
#     """
#     # Load the FAISS index
#     index = faiss.read_index(index_path)
    
#     # Load the metadata containing text chunks
#     with open(index_path + '.metadata', 'rb') as f:
#         metadata = pickle.load(f)
#     chunks = metadata['chunks']
    
#     # Initialize the model and generate embedding for query
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = model.encode([query])
    
#     # Convert to correct format and type
#     query_embedding = np.array(query_embedding).astype('float32')
    
#     # Search the index
#     distances, indices = index.search(query_embedding, k)
    
#     # Get the corresponding text chunks
#     similar_chunks = [chunks[idx] for idx in indices[0]]
    
#     return similar_chunks

# # Configure the assistant
# assistant_config = {
#     "name": "RAG_TCS_Assistant",
#     "system_message": """You are a helpful assistant who can answer questions based on provided context.
#     If you cannot find the answer in the context, say "I cannot answer based on the available information."
#     Always use the context to provide accurate answers.""",
#     "llm_config": {
#         "config_list": [{"model": "llama3"}],
#         "temperature": 0.7,
#         "base_url": "http://localhost:11434",
#         "api_type": "ollama"
#     }
# }

# # Create the agents
# assistant = autogen.AssistantAgent(**assistant_config)

# user_proxy = autogen.UserProxyAgent(
#     name="User",
#     system_message="A human user who asks questions.",
#     code_execution_config=False
# )

# def get_rag_response(query: str) -> str:
#     # Get relevant context from vector DB
#     context_chunks = search_similar_chunks(query, "VectorDb/faiss_index.db")
#     context = "\n\n".join(context_chunks)
    
#     # Prepare the message with context
#     message = f"""Please answer this question: {query}
    
#     Use this context to formulate your answer:
#     {context}"""
    
#     # Initialize a new chat
#     user_proxy.initiate_chat(
#         assistant,
#         message=message
#     )
    
#     # Get the last response from the assistant
#     chat_history = user_proxy.chat_messages[assistant]
#     return chat_history[-1]["content"]

# if __name__ == "__main__":
#     print("Hello! Ask me anything (type 'bye' to exit)")
    
#     while True:
#         print("You:", end=" ")
#         user_question = input()
        
#         if user_question.lower() == "bye":
#             print("Goodbye!")
#             break
            
#         answer = get_rag_response(user_question)
#         print("\nAssistant:", answer, "\n")




import autogen
import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def search_similar_chunks(query: str, index_path: str, k: int = 3) -> list[str]:
    """
    Search for chunks similar to the query in the FAISS index
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
    """
    # Join context chunks into single string
    context = "\n\n".join(context_chunks)
    
    # Initialize Ollama model
    llm = Ollama(model="llama2")
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use the following context to answer the question. 
                      If you cannot find the answer in the context, say "I cannot answer based on the provided context."
                      \n\nContext:\n{context}"""),
        ("user", "Question: {question}")
    ])
    
    # Create chain and get response
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })
    
    return response

# Configure the assistant
config_list = [
    {
        "model": "llama2",
        "base_url": "http://localhost:11434",
        "api_key": "ollama"
    }
]

assistant_config = {
    "name": "Assistant",
    "llm_config": {
        "config_list": config_list,
        "temperature": 0
    },
    "system_message": """You are a helpful AI assistant. You will receive questions along with relevant context.
    Use the context to provide accurate answers. If you cannot find the answer in the context, say so clearly."""
}

# Create the agents
assistant = autogen.AssistantAgent(**assistant_config)

user_proxy = autogen.UserProxyAgent(
    name="User",
    system_message="A human user who asks questions.",
    code_execution_config=False
)

def get_rag_response(query: str) -> str:
    # Get relevant context from vector DB
    context_chunks = search_similar_chunks(query, "VectorDb/faiss_index.db")
    context = "\n\n".join(context_chunks)
    
    # Prepare the message with context
    message = f"""Please answer this question: {query}
    
    Use this context to formulate your answer:
    {context}"""
    
    # Initialize a new chat
    user_proxy.initiate_chat(
        assistant,
        message=message
    )
    
    # Get the last response from the assistant
    chat_history = user_proxy.chat_messages[assistant]
    return chat_history[-1]["content"]

if __name__ == "__main__":
    print("Hello! Ask me anything (type 'bye' to exit)")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    while True:
        print("You:", end=" ")
        user_question = input()
        
        if user_question.lower() == "bye":
            print("Goodbye!")
            break
            
        answer = get_rag_response(user_question)
        print("\nAssistant:", answer, "\n")
