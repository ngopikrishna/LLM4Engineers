# https://www.facebook.com/reel/934750531820047\


import sagemaker, boto3, json
from botocore.config import Config

def ask_claude_sonnet(context, query):
    LLM_MODEL = 'anthropic.claude-3-sonnet-20240229-v1:0' 

    MAX_TOKENS = 10000
    TOP_P = 0.0000000000000000000001
    ANTHROPIC_VERSION = "bedrock-2023-05-31"

    answer_prompt_template = """Answer the question as truthfully as possible using the following key points in the context. 
    Instructions:
    -Use all information provided in the context. Do not cut your answer short or miss anything provided in the key points. 
    -Only use relevant sentences from the provided context that are absolutely required answer the following question. 
    -Your response should exactly answer the question and use exact texts extracted from context without making any modifications. 
    -If you believe the question cannot be answered from the given context, return the phrase "Sorry, I don't know".

    Context:
    Key points

    {context}


    {query}
    """

    boto_config = Config(read_timeout=1000)
    bedrock = boto3.client(service_name="bedrock-runtime", config=boto_config, region_name='us-east-1')
    system_message = answer_prompt_template.replace("{context}", context)
    system_message = system_message.replace("{query}", query)
 
 
    native_request = {
        "max_tokens": MAX_TOKENS,
        "anthropic_version": ANTHROPIC_VERSION,
        "top_p":TOP_P,
        "messages":  [
            {
                "role": "user",
                "content": [{"type": "text", "text": system_message}],
            }
        ],
    }
 
    request = json.dumps(native_request)
    response = bedrock.invoke_model(body=request, modelId=LLM_MODEL)
    response_body = json.loads(response.get("body").read())
    output = response_body.get("content")[0]['text']
    return output




chunks = """
10. Gopi completed his bachelors in engineering from Andhra University
11. Gopi is the father of Dheeraj
12. Gopi's first book is named "Machine Learning for Engineers"
13. Gopi's wife is Padma Latha
14. Gopi's father is a retired engineer
15. Gopi's mother is a home maker
16. Gopi's first job is at TCS
17. Gopi has double masters degrees from SUNY, Buffalo and from Amrita University
"""

query1 = """What is Gopi's occupation?"""
query2 = """What are Gopi's hobbies?"""

response = ask_claude_sonnet(chunks, query1)
print(response)