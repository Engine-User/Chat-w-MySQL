from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
import os
from urllib.parse import quote_plus
from langchain_groq import ChatGroq

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    # Properly encode the password
    encoded_password = quote_plus(password)
    db_uri = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Which 3 items have been ordered the most?
    SQL Query: SELECT product_name, SUM(quantity) as total_ordered FROM orders GROUP BY product_name ORDER BY total_ordered DESC LIMIT 3;
    Question: List 10 customer names
    SQL Query: SELECT customer_name FROM customers LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response as well as a SQL command/query for the user to run(if applicable).
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! This is an SQL assistant for your database. Connect your Database and ask anything about the data."),
    ]

load_dotenv()

# Set dark theme
st.set_page_config(page_title="Chat with MySQL Database", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Graduate&display=swap');
    
    body {
        color: #C0C0C0;
        font-family: 'Graduate', sans-serif !important;
    }
    
    .stButton>button {
        color: #C0C0C0;
        background-color: #2b2b2b;
        font-family: 'Graduate', sans-serif !important;
    }
    
    .stTextInput>div>div>input {
        color: #C0C0C0;
        font-family: 'Graduate', sans-serif !important;
    }
    
    .stMarkdown {
        color: #C0C0C0;
        font-family: 'Graduate', sans-serif !important;
    }

    /* Apply Graduate font to all elements */
    * {
        font-family: 'Graduate', sans-serif !important;
    }

    /* Ensure that code blocks and pre-formatted text use a monospace font */
    pre, code {
        font-family: monospace !important;
    }

    /* Custom styles for the header */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .header-line {
        width: 80%;
        height: 2px;
        background-color: #C0C0C0;
        margin: 10px 0;
    }

    .header-title {
        font-size: 2.5em;
        color: #C0C0C0;
        text-align: center;
        padding: 10px 20px;
        background-color: #2b2b2b;
        border-radius: 10px;
        transition: transform 0.3s, box-shadow 0.3s;
    }

    .header-title:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(192, 192, 192, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Custom HTML for the header
st.markdown("""
<div class="header-container">
    <div class="header-line"></div>
    <h1 class="header-title">Chat with MySQL Database</h1>
    <div class="header-line"></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    # About button moved to the top
    if st.button("About"):
        st.markdown("This app allows you to chat with your MySQL database using natural language. Made by <span style='color: red;'>Engineer</span>.", unsafe_allow_html=True)

    st.subheader("Database Connection Settings")
    st.write("Connect to your MySQL database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", key="Password")
    st.text_input("Database", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Failed to connect to the database: {str(e)}")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Ask about your database...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        if "db" not in st.session_state:
            st.error("Please connect to the database first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
