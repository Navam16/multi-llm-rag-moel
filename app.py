import streamlit as st
import os
import uuid

# ==========================================
# 1. APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="OmniAgent Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f9fafb; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background-color: #f0fdf4;
        border: 1px solid #dcfce7;
        border-radius: 9999px;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API SETUP
# ==========================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("🚨 Secrets not found!")
    st.stop()

# Imports
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# SAFELY IMPORT DUCKDUCKGO
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    ddg_available = True
except ImportError:
    ddg_available = False
    st.warning("⚠️ Web Search tool could not be loaded. Continuing without it.")

# ==========================================
# 3. SIDEBAR: MODEL SELECTION
# ==========================================
with st.sidebar:
    st.title("🤖 OmniAgent")
    st.caption("Research Suite")
    st.divider()

    st.markdown('**🧠 Select AI Model**')
    model_option = st.selectbox(
        "Choose Model",
        [
            "Llama 3.1 70B (via Groq)",
            "Llama 3.1 8B (via Groq)",
            "Gemini 1.5 Pro (Google)",
            "GPT-4o (OpenAI)",
            "Mixtral 8x7B (via Groq)"
        ]
    )

    st.markdown('**📂 Knowledge Base**')
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ==========================================
# 4. LLM BACKEND
# ==========================================
def get_llm_instance(choice):
    if "Llama 3.1 70B" in choice:
        return ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY, temperature=0)
    elif "Llama 3.1 8B" in choice:
        return ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)
    elif "Mixtral" in choice:
        return ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY, temperature=0)
    elif "GPT-4o" in choice:
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    elif "Gemini" in choice:
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0)
    return ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

@st.cache_resource
def setup_agent(model_choice, pdf_data=None):
    # Initialize Tools List
    tools = [
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    ]
    
    # Add Web Search only if available
    if ddg_available:
        try:
            tools.append(DuckDuckGoSearchRun(name="web_search"))
        except Exception:
            pass # Skip if it fails at runtime
    
    # Add RAG Tool if PDF uploaded
    if pdf_data:
        with open("temp_rag.pdf", "wb") as f:
            f.write(pdf_data.getbuffer())
        loader = PyPDFLoader("temp_rag.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever()
        
        @tool
        def query_pdf(question: str):
            """Use this to answer questions about the uploaded PDF file."""
            results = retriever.invoke(question)
            return "\n\n".join([doc.page_content for doc in results])
        tools.append(query_pdf)

    llm = get_llm_instance(model_choice)
    llm_with_tools = llm.bind_tools(tools)
    
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    def reasoner(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
    graph = StateGraph(State)
    graph.add_node("agent", reasoner)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())

agent_app = setup_agent(model_option, uploaded_file)

# ==========================================
# 5. CHAT UI
# ==========================================
st.markdown("""
<div style="margin-bottom: 20px;">
    <div class="status-badge">
        <span style="width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%; margin-right: 8px;"></span>
        Agent Active
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask OmniAgent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        with st.spinner(f"Thinking with {model_option}..."):
            events = agent_app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="values"
            )
            full_response = ""
            for event in events:
                if "messages" in event:
                    full_response = event["messages"][-1].content
                    message_placeholder.markdown(full_response)
            
            if not full_response:
                full_response = "✅ Task completed."
                message_placeholder.markdown(full_response)
                
    st.session_state.messages.append({"role": "assistant", "content": full_response})
