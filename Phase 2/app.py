import streamlit as st

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms import Ollama


# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(
    page_title="AI Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# Custom Styling
# ---------------------------

st.markdown(
    """
    <style>
        .main {
            padding-top: 2rem;
        }

        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
        }

        .title {
            text-align:center;
            font-size:42px;
            font-weight:bold;
            margin-bottom:10px;
        }

        .subtitle {
            text-align:center;
            color:gray;
            margin-bottom:30px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Header
# ---------------------------

st.markdown(
    '<div class="title">🤖 AI-Powered Q&A System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">LangChain + Ollama + Qwen2.5 + Buffer Memory</div>',
    unsafe_allow_html=True
)

# ---------------------------
# Session State
# ---------------------------

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "conversation" not in st.session_state:
    llm = Ollama(
        model="qwen2.5:7b",
        temperature=0.7
    )

    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()

    st.write("### Memory Type")
    st.info("ConversationBufferMemory")

    st.write("### Model")
    st.success("qwen2.5:7b")

# ---------------------------
# Display Chat History
# ---------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# User Input
# ---------------------------

user_question = st.chat_input("Ask me anything...")

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        with st.spinner("🤖 Thinking..."):

            response = st.session_state.conversation.predict(
                input=user_question
            )

            for word in response.split():
                full_response += word + " "
                response_placeholder.markdown(
                    full_response + "▌"
                )

            response_placeholder.markdown(
                full_response
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )