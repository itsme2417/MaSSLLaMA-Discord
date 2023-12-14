
# MaSSLLaMA-Discord
Warning before you go further: The code is probably of extremely poor quality. Much of this was written many months ago so most things are handled in a relatively outdated way.

This is a multimodal LLM based Discord 'userbot' that supports image input, image generation, basic internet search and unmaintained and basic TTS.
It is based around TabbyAPI or llama.cpp's server however it can possibly work with other OpenAI compatible endpoints while in the tabbyapi mode.

## Usage

Note: 
While it should work fine for basic usage with most models, it has been mostly used with `airoboros-l2-70b-3.1`. It works hit or miss with `mixtral 8x7b-instruct` with it repeatedly mentioning it isnt actually able to generate images, despite doing so.

install the requirements:
`pip install -r requirements.txt`

rename config.json.example to config.json then change the options as required.

## Config File 

Here's a brief explanation of the options in the config file:

- `Backend`: The backend that runs the LLM. Options: `tabbyapi` or `llama.cpp`
- `HOST` and `PORT`: The IP address and port of the backend.
- `Whitelisted_servers`: A list of servers and channels where the bot is allowed to operate. The format is `[server id, channel id/0 for any]`.
- `Base_Context`: The system prompt for the model. `GU912LD` will be replaced for the discord guild name, while `{daterplc}` will be replaced with the current date.
- `model_Name`: The username of the model.
- `token`: The Discord user token.
- `api_key`: The API token for the TabbyAPI server.
- `admin_id`: The Discord ID of the admin user.

## Commands

Here are the available commands and their functions:

- `?clearmem`: Clears the context.
- `?raw`: Makes messages be sent directly to comfyui when generating images.
- `?joinvc`: Makes the bot join a voice channel.
- `?setinit`: Sets the system prompt for the model, format is `?setinit {<*systemprompt*`
- `?reloadmem`: Reloads the context of the bot from the saved file.
- `?lobotomy`: Removes the last two messages from the context.
- `?enableimg`: Enables image generation for the bot.
- `?disableimg`: Disables image generation for the bot.
- `?save`: Saves the current context to a file.
- `?block`: Blocks a user from interacting with the bot. Might or might not work, usage is `?block @user`
- `?clearblocklist`: Clears the block list.
- `?reloadconfig`: Reloads the configuration file.

## Donations

Patreon: https://www.patreon.com/llama990
LTC: Le23XWF6bh4ZAzMRK8C9bXcEzjn5xdfVgP
If you want to mess around with the bot, im currently running it on the following discord server:
https://discord.gg/zxPCKn859r
