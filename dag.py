from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import SC_PROMPT
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import RedisChatMessageHistory
from operator import itemgetter
import os
import uuid
from dotenv import load_dotenv

load_dotenv(dotenv_path='.venv/.env')
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGHOST"),
    port=int(os.environ.get("PGPORT")),
    database=os.environ.get("PGDATABASE"),
    user=os.environ.get("PGUSER"),
    password=os.environ.get("PGPASSWORD"),
)
EMBEDDINGS = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_version='2023-09-01-preview'
)
NAMESPACE = "pgvector/cbot_corpus"
COLLECTION_NAME = 'CBOT-CORPUS'
REDIS_URL = os.getenv('REDIS_URL')
vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=EMBEDDINGS,
)
retriever = vectorstore.as_retriever()
USER_ID = str(uuid.uuid4())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = AzureChatOpenAI(
    azure_deployment="gpt-4-1106",
    openai_api_version="2023-09-01-preview",
)

chain = (
    ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("system", """Given the user question below, classify it as either being about `Social Care Chatbot`, `Council Chatbot`.

Do not respond with more than one word. If there is not enough information to determine the worker, choose the worker that was used"""),
            ("human", "{question}"),
        ]
    )
    | llm
    | StrOutputParser()
)

social_care_chain = (
    ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("system", SC_PROMPT),
            ("human", "{question}"),
        ]
    )
    | llm
    | StrOutputParser()
)

council_chain = (
    {"context": itemgetter("question") | retriever, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
    | PromptTemplate.from_template("""
    History: {history}
    You are the Cura, a Council Chatbot for the Wigan Council. Your role is to provide the human with answers to queries pertaining to the council. Use the following pieces of retrieved context by the assistant to answer the human's questions. Do not answer questions irrelevant to the council. If you don't know the answer or the answer is not provided in the retrieved context, inform that you don't know and is irrelevant. Provide as much detail as possible in your answer. Do not make anything up. If the human's message is a greeting, Greet them and provide some questions they can ask you from the given context(EXAMPLE: Here are some questions you can ask me:). If you are recommending the human to a website, provide the link as a hyperlink. Write your responses in Markdown.
    Context: {context}
    Question: {question}

    Helpful Answer:
    """)
    | ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("system",
                 "You are the Cura, a Council Chatbot for the Reading Borough Council. Your role is to provide the human with answers to queries pertaining to the council. Use the following pieces of retrieved context by the assistant to answer the human's questions. Do not answer questions irrelevant to the council. If you don't know the answer or the answer is not provided in the retrieved context, inform that you don't know and is irrelevant. Provide as much detail as possible in your answer. Do not make anything up. If the human's message is a greeting, Greet them and provide some questions they can ask you from the given context(EXAMPLE: Here are some questions you can ask me:). If you are recommending the human to a website, provide the link as a hyperlink. Write your responses in Markdown."),
                MessagesPlaceholder(variable_name="context"),
                ("human", "{question}"),
            ]
        )
    | llm
    | StrOutputParser()
)
general_chain = (
    ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            (
                "system",
                "Respond to the Greeting",
            ),
            ("human", "{question}"),
        ]
    )
    | llm
    | StrOutputParser()

)

branch = RunnableBranch(
    (lambda x: "social care chatbot" in x["topic"].lower(), social_care_chain),
    (lambda x: "council chatbot" in x["topic"].lower(), council_chain),
    council_chain,
)
# runnable_branch = RunnableWithMessageHistory(
#     branch,
#     lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
#     input_messages_key="question",
#     history_messages_key="history"
# )
full_chain = {"topic": chain, "question": lambda x: x["question"], "history": lambda x: x["history"]}| branch
final_runnable = RunnableWithMessageHistory(
    full_chain,
    lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
    input_messages_key="question",
    history_messages_key="history"
)
# print(final_runnable.get_graph().print_ascii())
while True:
    question = str(input('\n\nenter message: '))
    if 'history' in question:
        print(final_runnable.get_session_history(USER_ID))
    else:
        # print(question)
        print(final_runnable.invoke({"question": question},
                                config=RunnableConfig(configurable={"session_id": USER_ID})))
