import ollama


modelname = 'llama3'
messages=[ {'role': 'user', 'content': 'Why is the sky blue?',},]
response = ollama.chat(model=modelname, messages=messages)
print(response['message']['content'])

