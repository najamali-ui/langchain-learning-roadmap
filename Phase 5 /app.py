import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import get_stock_price, get_weather, get_faq_retriever_tool

# Page Configuration
st.set_page_config(page_title="TechStore AI Support", page_icon="🤖", layout="centered")

st.title("🤖 TechStore AI Support Chatbot")
st.write("Ask me about store policies, the weather, or check the latest stock prices!")

# Initialize Memory
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

# Initialize Agent
@st.cache_resource
def initialize_agent():
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
    
    faq_tool = get_faq_retriever_tool()
    tools = [get_stock_price, get_weather, faq_tool]
    
    system_prompt = '''You are a helpful customer support chatbot for TechStore.
You can answer questions using the provided tools.
Always try to use the techstore_faq_search tool first if the user asks about policies, returns, or the store.
If the user asks for stock prices, use get_stock_price.
If the user asks for weather, use get_weather.'''
    
    agent = create_react_agent(llm, tools, prompt=system_prompt, checkpointer=st.session_state.memory)
    return agent

agent_executor = initialize_agent()

# Display Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle User Input
if user_prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                config = {"configurable": {"thread_id": "techstore_session"}}
                response = agent_executor.invoke({"messages": [("user", user_prompt)]}, config)
                output = response["messages"][-1].content
            except Exception as e:
                output = f"I encountered an error: {e}"
        
        st.write(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
