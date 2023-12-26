import json
import os
import random
from pathlib import Path

import requests
import transformers


class Config:
    def __init__(self):
        script_dir = Path(os.path.abspath(__file__)).parent

        with open(os.path.join(script_dir, 'config.json')) as config_file:
            config = json.load(config_file)

        self.backend = config['Backend']
        self.host = config['HOST']
        self.port = config['PORT']
        self.whitelisted_servers = config['Whitelisted_servers']
        self.base_context = config['Base_Context']
        self.model_name = config['model_Name']
        self.enabled_features = config['Enabled_features']
        self.llm_parameters = config['LLM_parameters']
        self.adminid = config['admin_id']
        self.token = config['token']
        self.api_key = config['api_key']
        self.ctxlen = config['max_seq_len']
        self.reservespace = config['reserve_space']

        print("Loaded config")

config = Config()
API_ENDPOINT_URI = f"http://{config.host}:{config.port}/"
API_KEY = config.api_key 

TABBY = False
if config.backend == "tabbyapi":
    TABBY = True
if TABBY:
    API_ENDPOINT_URI += "v1/completions"
else: 
    API_ENDPOINT_URI += "completion"

def tokenize(input):
    if TABBY:
        payload = {
        "add_bos_token": "true",
        "encode_special_tokens": "true",
        "decode_special_tokens": "true",
        "text": input
        }
        request = requests.post(
            API_ENDPOINT_URI.replace("completions", "token/encode"),
            headers={"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=360,
        )
        return request.json()
    else:
        payload = {
        "content": input
        }
        request = requests.post(
            API_ENDPOINT_URI.replace("completion", "tokenize"),
            json=payload,
            timeout=360,
        )
        return {'length': len(request.json()['tokens']) }

def infer(prmpt, system='', temperature=config.llm_parameters['temperature'], username="", bsysep="<</SYS>>", esysep="", modelname="", eos="</s><s>",beginsep="[INST]",endsep="[/INST]", mem=[], few_shot="", max_tokens=250, stopstrings=[], top_p=config.llm_parameters['top_p'], top_k=config.llm_parameters['top_k']):
    content = ''
    memory = mem
    prompt = f"{bsysep}\n"+ system + f"{esysep}\n" + few_shot + "".join(memory) + f"\n{beginsep}{username}{prmpt}{endsep}\n{beginsep}{modelname}\n" 
    #This feels wrong.

    print(f"Token count: {tokenize(prompt)['length']}")
    removal = 0
    reserved_space = max_tokens
    if not config.reservespace:
        reserved_space = 0
    while tokenize(prompt)['length'] + reserved_space > config.ctxlen and len(memory) > 2:
        print(f"Removing old memories: Pass:{removal}")
        removal+=1
        memory = memory[removal:]
        prompt = f"{bsysep}\n"+ system + f"{esysep}\n" + few_shot + "".join(memory) + f"\n{beginsep}{username}\n{prmpt}{endsep}\n{beginsep}{modelname}\n" 

    payload = {
            "prompt": prompt,
            "model": "gpt-3.5-turbo-instruct",
            "max_tokens": max_tokens,
            "n_predict": max_tokens,
            "stream": True,
            "seed": random.randint(1000002406736107, 3778562406736107), #Was acting weird without this
            "top_k": top_k,
            "top_p": top_p,
            "min_p": config.llm_parameters['min_p'],
            "stop": [beginsep] + stopstrings,
            "temperature": temperature,
        }

    request = requests.post(
        API_ENDPOINT_URI,
        headers={"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        json=payload,
        stream=True,
        timeout=360,
    )

    if request.encoding is None:
        request.encoding = "utf-8"

    for line in request.iter_lines(decode_unicode=True):
        if line:
            if TABBY:
                print(json.loads(" ".join(line.split(" ")[1:]))['choices'][0]['text'], end="", flush=True)

                content += json.loads(" ".join(line.split(" ")[1:]))['choices'][0]['text']
            else:
                try:
                    if "data" in line:
                        print(json.loads(" ".join(line.split(" ")[1:]))['content'], end="", flush=True)

                        content += json.loads(" ".join(line.split(" ")[1:]))['content']           
                except Exception as e:
                    print(e)
    print("")
    memory.append(f"\n{beginsep} {username} {prmpt.strip()}\n{endsep} {modelname} {content.strip()}{eos}")
    return [content,memory]