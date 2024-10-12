from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import os
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

def get_mock_database():
    # This function returns a mock database schema
    return """
    CREATE TABLE customers (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(100),
        created_at TIMESTAMP
    );

    CREATE TABLE orders (
        id INT PRIMARY KEY,
        customer_id INT,
        product_name VARCHAR(100),
        quantity INT,
        order_date TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    """

def get_sql_chain():
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
    SQL Query: SELECT name FROM customers LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    def get_schema(_):
        return get_mock_database()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
def get_response(user_query: str, chat_history: list):
    sql_chain = get_sql_chain()
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking questions about the company's database.
    Based on the table schema below, question, and SQL query, write a natural language response explaining the query and what it would do if executed.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}

    Provide a detailed explanation of the SQL query and how it addresses the user's question. Also, mention that this is a demonstration and the query is not actually being executed on a real database.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: get_mock_database(),
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
        AIMessage(content="Hello! This is an SQL assistant for a mock database. Ask anything about the data, and I'll generate SQL queries and explain them."),
    ]

# Set dark theme
st.set_page_config(page_title="Chat with Mock SQL Database", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling (unchanged)
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

    * {
        font-family: 'Graduate', sans-serif !important;
    }

    pre, code {
        font-family: monospace !important;
    }

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
    <h1 class="header-title">Chat with Mock SQL Database</h1>
    <div class="header-line"></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.button("About"):
        st.markdown("This app demonstrates how to chat with a mock SQL database using natural language. It generates SQL queries based on your questions but doesn't execute them. Made by <span style='color: red;'>Engineer</span>.", unsafe_allow_html=True)

    st.subheader("Mock Database Schema")
    st.code(get_mock_database(), language="sql")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Ask about the mock database...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            try:
                response = get_response(user_query, st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
