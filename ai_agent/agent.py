import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent

from ai_agent.tools import turbine_maintenance_reports_predictor, turbine_maintenance_predictor, turbine_specifications_retriever
from ai_agent.agent_instructions import system_template

openAI_api_key = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=openAI_api_key)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def interact_with_agent(query: str) -> str:
    """
    Interact with the agent to process a query and return the response.

    Args:
        query (str): The user query to interact with the agent.

    Returns:
        str: The agent's response to the query.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_template),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{query}'),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ]
    )

    tools = [turbine_maintenance_reports_predictor, turbine_maintenance_predictor, turbine_specifications_retriever]

    agent = create_openai_functions_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False
    )

    raw_response = agent_executor.invoke({
        'query': query
    })

    return raw_response['output']