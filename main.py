from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable.config import RunnableConfig
import operator
from typing import Annotated, Sequence, TypedDict
import functools
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from utils import SC_PROMPT
import os
import uuid
from dotenv import load_dotenv

load_dotenv(dotenv_path='.venv/.env')

REDIS_URL = os.getenv('REDIS_URL')
USER_ID = str(uuid.uuid4())

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


def create_agent(llm: AzureChatOpenAI, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    # Set up a chain of operations
    chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
        input_messages_key="messages",
        history_messages_key="history",
    )


def agent_node(state, agent, name):
    result = agent.invoke(state, config=RunnableConfig(configurable={"session_id": USER_ID}))
    return {"messages": [AIMessage(content=result, name=name)]}


def main():
    print('check 1')
    members = ["Council Chatbot", "Social Care Chatbot", "Greeting bot"]
    system_prompt = (
        " You are a supervisor tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task once and respond with their results and status. Once a worker has responded, FINISH. When finished,"
        " respond with FINISH."
    )
    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = ["FINISH"] + members
    # Using openai function calling can make output parsing easier for us
    function_def = {
        "name": "route",
        "description": "Select the next worker or FINISH.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                " Given the conversation above, who should act next or choose FINISH? "
                " If the question has been answered then choose FINISH."
                " If the worker has answered and the user is required to answer then choose FINISH. "
                " If the worker has replied with a question then choose FINISH. "
                " If the either worker has replied to the current query then choose FINISH. "
                " Always choose FINISH after you have chosen a Worker once."
                " If there is not enough information to determine the worker, choose the worker that was used last."
                " Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4-1106",
        openai_api_version="2023-09-01-preview",
    )

    supervisor_chain = (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
    )
    supervisor_runnable = RunnableWithMessageHistory(
        supervisor_chain,
        lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
        input_messages_key="messages",
        history_messages_key="history",
    )
    print('check 2')
    research_agent = create_agent(llm,  "You are the Cura, a Council Chatbot for the Reading Borough Council. Your role is to provide the human with answers to queries pertaining to the council.")
    research_node = functools.partial(agent_node, agent=research_agent, name="Council Chatbot")

    # NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
    code_agent = create_agent(
        llm,
        SC_PROMPT,
    )
    code_node = functools.partial(agent_node, agent=code_agent, name="Social Care Chatbot")
    greeting_agent = create_agent(
        llm,
        'You are the Greeting bot, you reply to greetings.',
    )
    greeting_node = functools.partial(agent_node, agent=greeting_agent, name="Greeting bot")
    print('check 3')
    workflow = StateGraph(AgentState)
    workflow.add_node("Council Chatbot", research_node)
    workflow.add_node("Social Care Chatbot", code_node)
    workflow.add_node("Greeting bot", greeting_node)
    workflow.add_node("supervisor", supervisor_chain)
    for member in members:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    print('check 4')
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # Finally, add entrypoint
    workflow.set_entry_point("supervisor")
    print('check 5')

    graph = workflow.compile()
    print('check 6')
    print(graph.get_graph().print_ascii())
    while True:
        for s in graph.stream(
                {
                    "messages": [
                        HumanMessage(content=str(input()))
                    ]
                }, config=RunnableConfig(configurable={"session_id": USER_ID})
        ):
            if "__end__" not in s:
                print(s)
                print("----")
        print('check 7')

if __name__ == '__main__':
    main()
