import base64
import json
import random
import urllib.parse
import urllib.request
import uuid
import os
import websocket
from pathlib import Path

client_id = str(uuid.uuid4())
with open(os.path.join(Path(os.path.abspath(__file__)).parent, 'comfyui_workflow_lcm.json')) as workflow:
    prompt_text_lcm = json.load(workflow)
with open(os.path.join(Path(os.path.abspath(__file__)).parent, 'comfyui_workflow_turbovision.json')) as workflow:
    prompt_text_turbovision = json.load(workflow)

    
def queue_prompt(prompt, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id, server_address):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt, server_address):
    prompt_id = queue_prompt(prompt, server_address)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id, server_address)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'], server_address)
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

def generate(prmpt, server_address, seed = 0, width = 1024, height = 1024):
  
    prompt = prompt_text_turbovision 
    prompt["6"]["inputs"]["text"] = prmpt
    if not seed == 0: 
        prompt["3"]["inputs"]["seed"] = seed
    else:
        seeed = random.randint(2000002406736107, 3778562406736107)
        print(f"Seed: {seeed}")
        prompt["3"]["inputs"]["seed"] = seeed
    prompt["5"]["inputs"]["width"] = width
    prompt["5"]["inputs"]["height"] = height
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt, server_address)
    image = []
    for node_id in images:
        for image_data in images[node_id]:
            image.append(base64.b64encode((image_data)).decode('utf-8'))
    return image

            