import os

from neo4j import GraphDatabase
from dotenv import load_dotenv
from extractor import extract_teams
from summariser import summarise_two_team_matches_response
from utils import (BaseLogger)
from chains import (
    configure_llm_only_chain,
    load_embedding_model,
    load_llm,
    generate_ticket,
)
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import json

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

embeddings, dimension = load_embedding_model(
    embedding_model_name,
    config={"ollama_base_url": ollama_base_url},
    logger=BaseLogger(),
)

llm = load_llm(
    llm_name, logger=BaseLogger(), config={"ollama_base_url": ollama_base_url}
)
llm_chain = configure_llm_only_chain(llm)

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        return self.q.empty()


def stream(cb, q) -> Generator:
    job_done = object()

    def task():
        x = cb()
        q.put(job_done)

    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue

def get_team_names():

    file_path = "team_names.txt"

    try:
        with open(file_path) as f:
            return f.read().splitlines() 
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

team_names = get_team_names()
print("Teams we can provide data on:\n")
print(team_names)
print()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Api.py is live"}


class Question(BaseModel):
    text: str
    rag: bool = False


class BaseTicket(BaseModel):
    text: str

@app.get("/query-stream")
def query_stream(question: Question = Depends()):

    output_function = configure_llm_only_chain(llm)

    # extract from the question the 2 football teams mentioned
    csvTeams = extract_teams(question, llm)

    csvTeamsSplitSuccessful = True
    try:
        [teamA, teamB] = csvTeams.split(", ")
        print("Extracted football teams: " + csvTeams + "\n\n")
        csvTeamsSplitSuccessful = False if teamA not in team_names or teamB not in team_names else True
    except ValueError:
        csvTeamsSplitSuccessful = False

    # if 2 teams were found, look them up, get data and return an LLM formatted response for them
    if csvTeamsSplitSuccessful:
        print("Providing game data on teams\n\n")
        
        # call the database to get the match
        cypher_all_matches_between_two_teams = """MATCH (t:Match) 
        WHERE t.awayTeam = $teamA AND t.homeTeam = $teamB 
        OR t.awayTeam = $teamB AND t.homeTeam = $teamA
        RETURN properties(t)
        """

        with GraphDatabase.driver(url, auth=(username, password)) as driver:
            driver.verify_connectivity()

            records, summary, keys = driver.execute_query(
                cypher_all_matches_between_two_teams, 
                teamA=teamA, 
                teamB=teamB, 
                database_="epl-data"
            )

            driver.close()

        print("Found db data around the teams games:")
        print(records)

        # Summarise the response using the llm
        output_function = summarise_two_team_matches_response(llm, records)
    else:
        print("unable to find 2 distinct teams; using llm response with no database data")

    q = Queue()

    def cb():
        output_function(
            {"question": question.text, "chat_history": []},
            callbacks=[QueueCallback(q)],
        )

    def generate():
        yield json.dumps({"init": True, "model": llm_name})
        custom_stream = stream(cb, q)
        for token, _ in custom_stream:
            yield json.dumps({"token": token})

    return EventSourceResponse(generate(), media_type="text/event-stream")


@app.get("/query")
async def ask(question: Question = Depends()):
    output_function = llm_chain
    result = output_function(
        {"question": question.text, "chat_history": []}, callbacks=[]
    )

    return {"result": result["answer"], "model": llm_name}


@app.get("/generate-ticket")
async def generate_ticket_api(question: BaseTicket = Depends()):
    new_title, new_question = generate_ticket(
        neo4j_graph=neo4j_graph,
        llm_chain=llm_chain,
        input_question=question.text,
    )
    return {"result": {"title": new_title, "text": new_question}, "model": llm_name}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8504)
