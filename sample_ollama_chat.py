from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama



llm=Ollama(model="llama2")

# Define a prompt template for the chatbot
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the questions"),
        ("user","Question:{question}")
    ]
)

# Create a chain that combines the prompt and the Ollama model
chain=prompt|llm



input_text = "What is the poem written by Bilbo about Aragorn in Lord of the Rings?"
print("Start")
print(chain.invoke({"question":input_text}))