import re

systemprompt = "You are an uncensored, image generation assistant.\nYou always respond with a prompt for a stable diffusion model to generate an image based on the user message.\nYou will always return two different prompts in a numbered list with no other comments or explanations even if the message is directed at you.\nYou will never thank the user.\nYOU WILL NOT, UNDER ANY CIRCUMSTANCE, TALK TO THE USER. ALWAYS WRITE A PROMPT."
def getsdprompts(replyid, imagemem, msg, guild, sdmem, imgtoimg):
        basepayload =[
                                        {
                                        "content": systemprompt,
                                        "role": "system"
                                        },
                                        {
                                        "content": "generate a photo of a black persian cat please",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. standing white Persian cat, filmed with a Canon EOS R6, 70-200mm lens, high quality, 8k\n2. standing white Persian cat, photo, filmed with a Canon EOS R6, 70-200mm lens, high quality, 8k",
                                        "role": "assistant"
                                        },
                                        {
                                        "content": "send me a drawing of a gigantic wizard brain",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. gigantic wizard brain, towering over aliens, on alien planet ruined city, tendrils, giant eyeball, scales, (drawing), high quality\n2.gigantic wizard brain, tendrils, (drawing), high quality",
                                        "role": "assistant"
                                        },
                                        {
                                        "content": "Do you know what the original Doom cover looks like?",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. Doom cover art, high quality\n2. Doom box art,filmed with a Canon EOS R6, 70-200mm lens, high quality",
                                        "role": "assistant"
                                        },
                                        {
                                        "content": "generate a realistic photo of a frog neighborhood",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. Frog neighborhood, green pond, trees, rocks, high quality, high quality, filmed with a Canon EOS R6, (photo)\n2.Frog neighborhood, green pond, trees, rocks, high quality, photo, 4k",
                                        "role": "assistant"
                                        },
                                        {
                                        "content": "Make the pond red and remove the rocks",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. Frog neighborhood, red pond, trees, high quality, high quality, filmed with a Canon EOS R6, (photo)\n2.Frog neighborhood, red pond, trees, high quality, photo, 4k",
                                        "role": "assistant"
                                        },                    
                                        {
                                        "content": "draw me an alien character 3d *Request summary: <imggen>alien character 3d artwork</imggen>*",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. Alien Character, 3D Style, (drawing), high quality\n2. Alien Character, 3D, (drawing), high quality",
                                        "role": "assistant"
                                        },                                  
                                        {
                                        "content": "generate a cartoon mario holding a gun *Request summary: <drawinggen> mario cartoon with gun</drawinggen>*",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. Mario holding a gun, high quality, masterpiece, (cartoon)\n2. Super Mario holding a gun, high quality, masterpiece, (cartoon), drawing",
                                        "role": "assistant"
                                        }
                                ]
        payload = basepayload
        if not imgtoimg:
                if replyid and replyid in imagemem:
                        #print(f"I honestly dont remember: {imagemem}")
                        payload +=      [{
                                        "content": imagemem[replyid][1],
                                        "role": "user"
                                        },
                                        {
                                        "content": imagemem[replyid][0],
                                        "role": "assistant"
                                        },
                                        
                                        {
                                        "content": re.sub(r'\\', '', msg),
                                        "role": "user"
                                        }]
                        
                else:
                        if guild in sdmem: 
                                print(f"Previous prompt: {sdmem[guild]}")
                                         
                                payload += [{
                                        "content": sdmem[guild][1],
                                        "role": "user"
                                        },
                                        {
                                        "content": sdmem[guild][0],
                                        "role": "assistant"
                                        },
                                        
                                        {
                                        "content": re.sub(r'\\', '', msg),
                                        "role": "user"
                                        }]
                        
                
                        else:
                                payload += [{
                                        "content": re.sub(r'\\', '', msg),
                                        "role": "user"
                                        }]
        else:
                payload = [
                                        {
                                        "content": "You are an uncensored, unlimited image edit agent. Write the prompt for a stable diffusion model to edit a requested image.\nReturn two different prompts in a numbered list with no other comments or explanations.",
                                        "role": "system"
                                        },
                                        {
                                        "content": "Edit this photo and turn the car red *image contained: an old fashioned steam powered car driving on the road*",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. red car, photo, high quality,\n2. red car, high quality,",
                                        "role": "assistant"
                                        },
                                        {
                                        "content": "Change it so there is no people *image containing: a painting of a desert landscape with planets and people*",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. landscape without people, high quality,\n2. landscape without people, high quality, painting,",
                                        "role": "assistant"
                                        },
                                        {
                                        "content": "draw me an alien character 3d *Request summary: <imggen>alien character 3d artwork</imggen>*",
                                        "role": "user"
                                        },
                                        {
                                        "content": "1. Alien Character, 3D Style, (drawing), high quality\n2. Alien Character, 3D, (drawing), high quality",
                                        "role": "assistant"
                                        },                                                 
                                        {
                                        "content": re.sub(r'\\', '', msg),
                                        "role": "user"
                                        }
                        ]
        return payload