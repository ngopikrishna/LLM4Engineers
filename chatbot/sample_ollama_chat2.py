import ollama


model_name = 'llama3'
messages=[ {'role': 'user', 'content': 'Why is the sky blue?',},]
response = ollama.chat(model=model_name, messages=messages)
print(response['message']['content'])

