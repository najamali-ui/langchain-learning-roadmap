from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOllama(model="qwen", temperature=0.3)
memory = MemorySaver()
agent = create_react_agent(llm, tools=[], checkpointer=memory, state_modifier="You are a helpful bot.")
config = {"configurable": {"thread_id": "1"}}
response = agent.invoke({"messages": [("user", "Hello!")]}, config)
print(response["messages"][-1].content)
