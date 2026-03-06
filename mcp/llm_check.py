import boto3
import requests
import json

AWS_REGION = "us-west-2"
maxTokens = 10000
modelID= 'us.amazon.nova-pro-v1:0'

def llm_call(system_prompt:str, user_prompt:str) -> str:
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    INFERENCE_PARAMS = {"maxTokens": maxTokens, "topP": 0.9, "topK": 20, "temperature": 0.7}


    message_history = [{"role": "user", "content": [{"text": user_prompt}]}]    
    request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_history,
            "system": [{"text":system_prompt}],
            "inferenceConfig": INFERENCE_PARAMS,
        }

    response = client.invoke_model(body=json.dumps(request_body), modelId=modelID)
    response_body = json.loads(response.get("body").read())
    return response_body['output']['message']['content'][0]['text']




if __name__ == "__main__":
    system_prompt = "You are a helpful, laconic assistant that can answer questions and help with tasks."
    user_prompt = "What is the capital of France?"
    response = llm_call(system_prompt, user_prompt)
    print(response)