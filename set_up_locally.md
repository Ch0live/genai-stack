# Steps to set up repo
- Create a `.env` file from the environment template file `env.example`

Available variables:
| Variable Name        | Default value                     | Description                                                             |
| -------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| OLLAMA_BASE_URL      | http://host.docker.internal:11434 | REQUIRED - URL to Ollama LLM API                                        |
| NEO4J_URI            | neo4j://database:7687             | REQUIRED - URL to Neo4j database                                        |
| NEO4J_USERNAME       | neo4j                             | REQUIRED - Username for Neo4j database                                  |
| NEO4J_PASSWORD       | password                          | REQUIRED - Password for Neo4j database                                  |
| LLM                  | llama2                            | REQUIRED - Can be any Ollama model tag, or gpt-4 or gpt-3.5 or claudev2 |
| EMBEDDING_MODEL      | sentence_transformer              | REQUIRED - Can be sentence_transformer, openai or ollama                |
| OPENAI_API_KEY       |                                   | REQUIRED - Only if LLM=gpt-4 or LLM=gpt-3.5 or embedding_model=openai   |
| LANGCHAIN_ENDPOINT   | "https://api.smith.langchain.com" | OPTIONAL - URL to Langchain Smith API                                   |
| LANGCHAIN_TRACING_V2 | false                             | OPTIONAL - Enable Langchain tracing v2                                  |
| LANGCHAIN_PROJECT    |                                   | OPTIONAL - Langchain project name                                       |
| LANGCHAIN_API_KEY    |                                   | OPTIONAL - Langchain API key                                            |

## Get dependency
- Use a Venv. Set up using this guide
	- [Install packages in a virtual environment using pip and venv - Python Packaging User Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
	- Remember to switch your terminal to that venv using `.venv\Scripts\activate` (or for linux terminals `source .venv/Scripts/activate`)
- Run `pip install -r requirements.txt` in the project directory to pull all required dependencies. This didn't work for me and in the end I just installed them all individually, but if you can do it this way you can automate installing them all while you make a cuppa or something ☕

## LLM Configuration
MacOS and Linux users can use any LLM that's available via Ollama. Check the "tags" section under the model page you want to use on https://ollama.ai/library and write the tag for the value of the environment variable `LLM=` in the `.env` file.
All platforms can use GPT-3.5-turbo and GPT-4 (bring your own API keys for OpenAI models).

**MacOS**
Install [Ollama](https://ollama.ai) on MacOS and start it before running `docker compose up` using `ollama serve` in a separate terminal.

**Linux**
No need to install Ollama manually, it will run in a container as
part of the stack when running with the Linux profile: run `docker compose --profile linux up`.
Make sure to set the `OLLAMA_BASE_URL=http://llm:11434` in the `.env` file when using Ollama docker container.

To use the Linux-GPU profile: run `docker compose --profile linux-gpu up`. Also change `OLLAMA_BASE_URL=http://llm-gpu:11434` in the `.env` file.

**Windows**
Ollama now supports Windows. Install [Ollama](https://ollama.ai) on Windows and start it before running `docker compose up` using `ollama serve` in a separate terminal. Alternatively, Windows users can generate an OpenAI API key and configure the stack to use `gpt-3.5` or `gpt-4` in the `.env` file.

## Set up Neo4j
- Download Neo4j for desktop
- Delete starter db (it conflicts with some names)
- Install APOC plugin:
	- Go to Neo4J, your new database and then plugins. Click the drop down under APOC and choose Install and Restart
	- Check APOC is added successfully and wait for the database to restart. If it fails just try re running it (it took 2 attempts for me first time)

- In a terminal run `streamlit run loader.py --server.port=8502 --server.address=localhost`.
	- Navigate to [localhost:8504](http://localhost:8504/) and see a Hello World response and check it's working
	- Use the web interface to load Stack Overflow questions

## Set up the LLM
- In a terminal run `ollama serve` to start the local LLM
  - Alternatively see the `.env` file for how to use other remotely hosted models

## Run the services
- In a terminal run `python api.py`. 
	- Navigate to [localhost:8504](http://localhost:8504/) and see a Hello World response and check it's working
- In a terminal run `streamlit run bot.py --server.port=8501 --server.address=localhost`
	- Navigate to [localhost:8501](http://localhost:8501/) and see the UI to see it running
- In a terminal run `cd front-end; npm install; npm run dev` (swap semi colons for ampersands (&) if using windows terminal)