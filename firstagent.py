# import openai
# from openai import AzureOpenAI
# from azure.identity import EnvironmentCredential


# def get_access_token():
#     credential = EnvironmentCredential()
#     access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
#     return access_token.token



# # client = AzureOpenAI(api_version="2023-05-15", api_key=get_access_token(), base_url="https://cog-sandbox-dev-eastus2-001.openai.azure.com/")
# client = AzureOpenAI(api_version="2023-05-15", api_key=get_access_token(), base_url="https://cog-sandbox-dev-eastus2-001.openai.azure.com/")


# # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://cog-sandbox-dev-eastus2-001.openai.azure.com/")'
# # openai.api_base = "https://cog-sandbox-dev-eastus2-001.openai.azure.com/"


# model ='gpt-35-turbo-blue'
# messages= [{'role':'user', 'content':'tell me a joke about engineers'}]
# response = client.chat.completions.create(model=model, messages=messages)
# print(response.choices[0].message.content)

import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cog-sandbox-dev-eastus2-001.openai.azure.com/"

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version=os.environ["API_VERSION_GA"],
    )