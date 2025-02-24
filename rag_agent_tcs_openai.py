import autogen
import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from azure.identity import EnvironmentCredential

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

def __get_access_token():
    credential = EnvironmentCredential()
    access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return access_token.token

# Configure OpenAI settings
os.environ['AZURE_OPENAI_ENDPOINT'] = "https://cog-sandbox-dev-eastus2-001.openai.azure.com/"

config_list = [{
    "model": "gpt-35-turbo-blue",
    "api_type": "azure",
    "api_key": __get_access_token(),
    "api_version": "2023-05-15"
}]

# Configure the assistant
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
        message=message,
        silent=True
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
