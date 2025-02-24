import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO

#"GitamDemoKey"="LRzAoonVd3N_Ya7FXoTeENa9ig9V7Ve0eKOQm1w9jw"
def __get_access_token():
    credential = EnvironmentCredential()
    access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return access_token.token

if __name__=="__main__":

    # client = OpenAI(api_key="LRzAoonVd3N_Ya7FXoTeENa9ig9V7Ve0eKOQm1w9jw",
    #                 base_url="https://api.venice.ai/api/v1")

    # models = client.models
    # models_list = models.list()
    # for model in models_list.data:
    #     print(model.id)

    # print(client.auth_headers)


    # os.environ['AZURE_OPENAI_ENDPOINT'] = "https://cog-sandbox-dev-eastus2-001.openai.azure.com/"
    # client = AzureOpenAI(api_key=__get_access_token(), api_version="2023-05-15")
    
    # res = client.images.generate(prompt="An indian family of four enjoying a picnic on a beach",
    #                        model="fluently-xl",
    #                        n=1,
    #                        size="256x256",
    #                        quality="standard")
    # url1 = res["data"][0]["url"]
    # response = requests.get(url1)
    # # using the Image module from PIL library to view the image
    # Image.open(response.raw)







    import requests

    url = "https://api.venice.ai/api/v1/image/generate"

    payload = {
        "model": "fluently-xl",
        "prompt": "An indian family of four enjoying a picnic on a beach",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "hide_watermark": False,
        "return_binary": True,
        "seed": 123,
        "cfg_scale": 12,
        "style_preset": "3D Model",
        "negative_prompt": "<string>",
        "safe_mode": True
    }
    headers = {
        "Authorization": "Bearer LRzAoonVd3N_Ya7FXoTeENa9ig9V7Ve0eKOQm1w9jw",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    img = Image.open(BytesIO(response.content)) 
    img.show()