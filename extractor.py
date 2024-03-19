from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Function to extract from question the relevant clubs for querying knowledge graph
prompt = ChatPromptTemplate.from_template(
  (
      """
      You are an expert extraction algorithm tasked at extracting what football match (or matches) the user wants to know about. 
      You must return the 2 football clubs that the user has asked about from the question. 

      Return the 2 clubs in a comma separated list. For example

      Brighton & Hove Albion FC, Arsenal FC
      
      Football clubs must be referred to using one of the following names

      - Arsenal FC
      - Brighton & Hove Albion FC
      - Chelsea FC
      - Crystal Palace FC
      - Everton FC
      - Southampton FC
      - Watford FC
      - West Bromwich Albion FC
      - Manchester United FC
      - Newcastle United FC
      - AFC Bournemouth
      - Burnley FC
      - Leicester City FC
      - Liverpool FC
      - Stoke City FC
      - Swansea City AFC
      - Huddersfield Town AFC
      - Tottenham Hotspur FC
      - Manchester City FC
      - West Ham United FC

      For example, "The blues" should refer to Chelsea FC.

      If you do not know the value of an attribute asked to extract, return null for the attribute's value.

      The question: {question}
      """
  )
)

if __name__ == "__main__":
  load_dotenv(".env")

  key = os.getenv("OPENAI_API_KEY")
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key)

  chain = LLMChain(llm=llm, prompt=prompt)

  question = "What was the result between Man City and Arsenal?"
  print(chain.invoke(question))
  question = "Who won in Chelsea Bournemouth"
  print(chain.invoke(question))
  question = "Villa vs United?"
  print(chain.invoke(question))

# Cypher to get all matches between 2 teams
# MATCH (t:Match) 
# WHERE t.awayTeam = "Manchester United FC" AND t.homeTeam = "Arsenal FC" 
# OR t.awayTeam = "Arsenal FC" AND t.homeTeam = "Manchester United FC" 
# RETURN properties(t)
