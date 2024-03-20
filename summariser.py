from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os

from typing import List, Any

from langchain.prompts import (
    ChatPromptTemplate
)

class Match:
  matchDate: str
  winner: str

# Function to summarise the response to the user. Includes a summary of the users question and the relevant data
def summarise_two_team_matches_response(llm, records):
  
  # TODO: Code currently assumes 2 games. Refactor to handle any number of games
  # TODO: refactor to explain information about the game more clearly (might be best to deconstruct the records object before calling this method)

  system_message_prompt = ChatPromptTemplate.from_template(
    (
        """
        You have been asked to find the results of matches between the following Premier League football teams

        here is data about the games in a python dictionary format
        
        first game:

        {matchA}

        second game:
        
        {matchB}

        Summarise a response to the question using the data listed above  
        Only use data listed above. Do not infer any data outside of what is listed above.
        Separate each game into its own paragraph. Separate the paragraphs with 2 new lines.
        Return the game with the earlier match date first.
        Be unbiased in your response (you do not support either team).
        """
    )
  )

  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

  def generate_llm_output(user_input: str, callbacks: List[Any], prompt=chat_prompt) -> str:
    chain = prompt | llm
    answer = chain.invoke({"matchA": records[0], "matchB": records[1]}, config={"callbacks": callbacks}).content
    return {"answer": answer}
  
  # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
  # def generate_llm_output(
  #       user_input: str, callbacks: List[Any], prompt=chat_prompt
  #   ) -> str:
  #       chain = prompt | llm
  #       answer = chain.invoke(
  #           {"matchA": records[0], "matchB": records[1]}, config={"callbacks": callbacks}
  #       ).content
  #       return {"answer": answer}

  return generate_llm_output

if __name__ == "__main__":
  load_dotenv(".env")
  key = os.getenv("OPENAI_API_KEY")
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key)

  # print(summarise_two_team_matches_response("", llm))
