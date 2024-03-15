# GenAI Stack

Forked from docker/genai-stack repo. See [the set up locally guide](set_up_locally.md) for more guidance on local set up.


# Develop

> [!WARNING]
> There is a performance issue that impacts python applications in the `4.24.x` releases of Docker Desktop. Please upgrade to the latest release before using this stack.

# Applications

Here's what's in this repo:

| Name                  | Main files   | Compose name | URLs                  | Description                                                                                                   |
| --------------------- | ------------ | ------------ | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| Support Bot           | `bot.py`     | `bot`        | http://localhost:8501 | Main usecase. Fullstack Python application.                                                                   |
| Stack Overflow Loader | `loader.py`  | `loader`     | http://localhost:8502 | Load SO data into the database (create vector embeddings etc). Fullstack Python application.                  |
| PDF Reader            | `pdf_bot.py` | `pdf_bot`    | http://localhost:8503 | Read local PDF and ask it questions. Fullstack Python application.                                            |
| Standalone Bot API    | `api.py`     | `api`        | http://localhost:8504 | Standalone HTTP API streaming (SSE) + non-streaming endpoints Python.                                         |
| Standalone Bot UI     | `front-end/` | `front-end`  | http://localhost:8505 | Standalone client that uses the Standalone Bot API to interact with the model. JavaScript (Svelte) front-end. |

The database can be explored at http://localhost:7474.

## App 1 - Support Agent Bot

UI: http://localhost:8501
DB client: http://localhost:7474

- answer support question based on recent entries
- provide summarized answers with sources
- demonstrate difference between
    - RAG Disabled (pure LLM response)
    - RAG Enabled (vector + knowledge graph context)
- allow to generate a high quality support ticket for the current conversation based on the style of highly rated questions in the database.

![](.github/media/app1-rag-selector.png)
*(Chat input + RAG mode selector)*

|                                               |                                                   |
| --------------------------------------------- | ------------------------------------------------- |
| ![](.github/media/app1-generate.png)          | ![](.github/media/app1-ticket.png)                |
| *(CTA to auto generate support ticket draft)* | *(UI of the auto generated support ticket draft)* |

---

##  App 2 - Loader

UI: http://localhost:8502
DB client: http://localhost:7474

- import recent Stack Overflow data for certain tags into a KG
- embed questions and answers and store them in vector index
- UI: choose tags, run import, see progress, some stats of data in the database
- Load high ranked questions (regardless of tags) to support the ticket generation feature of App 1.




|                                  |                                   |
| -------------------------------- | --------------------------------- |
| ![](.github/media/app2-ui-1.png) | ![](.github/media/app2-model.png) |

## App 3 Question / Answer with a local PDF
UI: http://localhost:8503  
DB client: http://localhost:7474

This application lets you load a local PDF into text
chunks and embed it into Neo4j so you can ask questions about
its contents and have the LLM answer them using vector similarity
search.

![](.github/media/app3-ui.png)

## App 4 Standalone HTTP API
Endpoints: 
  - http://localhost:8504/query?text=hello&rag=false (non streaming)
  - http://localhost:8504/query-stream?text=hello&rag=false (SSE streaming)

Example cURL command:
```bash
curl http://localhost:8504/query-stream\?text\=minimal%20hello%20world%20in%20python\&rag\=false
```

Exposes the functionality to answer questions in the same way as App 1 above. Uses
same code and prompts.

## App 5 Static front-end
UI: http://localhost:8505

This application has the same features as App 1, but is built separate from
the back-end code using modern best practices (Vite, Svelte, Tailwind).  
The auto-reload on changes are instant using the Docker watch `sync` config.  
![](.github/media/app5-ui.png)
