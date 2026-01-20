import streamlit as st
import os
import uuid

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="OmniAgent Pro", page_icon="🤖", layout="wide")

# Custom CSS for status indicators
st.markdown("""
<style>
    .stApp { background-color: #f9fafb; }
    .success-box {
        padding: 10px; border-radius: 5px; background-color: #d1fae5;
        color: #065f46; border: 1px solid #34d399; margin-bottom: 10px;
    }
    .warning-box {
        padding: 10px; border-radius: 5px; background-color: #fef3c7;
        color: #92400e; border: 1px solid #f59e0b; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. API KEYS
# ==========================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("🚨 Secrets not found! Please check your Streamlit settings.")
    st.stop()

# ==========================================
# 3. IMPORTS
# ==========================================
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
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# ==========================================
# 4. SIDEBAR & PDF DIAGNOSTICS
# ==========================================
with st.sidebar:
    st.title("🤖 OmniAgent")
    st.caption("Research Suite")
    
    model_choice = st.selectbox("Choose Model", 
        ["Llama 3.1 8B (Groq - Fast)", "Llama 3.1 70B (Groq)", "GPT-4o (OpenAI)"])
    
    st.divider()
    st.markdown("**📂 Knowledge Base**")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    # --- PDF DIAGNOSTIC LOGIC ---
    pdf_processed = False
    
    if uploaded_file:
        with st.spinner("Analyzing PDF..."):
            try:
                # Save temp file
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load & Check text
                loader = PyPDFLoader("temp.pdf")
                raw_docs = loader.load()
                
                total_chars = sum([len(d.page_content) for d in raw_docs])
                
                if total_chars > 100:
                    st.success(f"✅ Loaded {len(raw_docs)} pages.")
                    pdf_processed = True
                else:
                    st.error("⚠️ PDF appears empty or is a scanned image (OCR needed).")
                    pdf_processed = False
            except Exception as e:
                st.error(f"❌ Error reading PDF: {e}")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

def get_llm(choice):
    if "8B" in choice:
        return ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    elif "GPT" in choice:
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    return ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)

# ==========================================
# 5. AGENT LOGIC
# ==========================================
@st.cache_resource(show_spinner=False)
def setup_agent(model_name, has_pdf):
    # Standard Tools
    tools = [
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    ]
    
    # Add PDF Tool ONLY if processing was successful
    if has_pdf:
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
        
        @tool
        def query_pdf(query: str):
            """Use this tool to answer questions about the uploaded PDF document."""
            results = retriever.invoke(query)
            # We explicitly label the data so the LLM knows it comes from the PDF
            formatted_results = "\n\n".join([f"Page {d.metadata.get('page', '?')}: {d.page_content}" for d in results])
            return f"SOURCE: UPLOADED PDF DOCUMENT\n\n{formatted_results}"
            
        tools.append(query_pdf)

    llm = get_llm(model_name)
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

# ==========================================
# 6. CHAT INTERFACE
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    agent = setup_agent(model_choice, pdf_processed)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # --- ENHANCED SYSTEM PROMPT FOR CITATIONS ---
    system_instruction = """You are a helpful research assistant. 
    IMPORTANT CITATION RULES:
    1. If you used the 'query_pdf' tool, you MUST end your answer with: 
       '**Source:** Uploaded PDF Document'
    2. If you used Wikipedia, you MUST end your answer with:
       '**Source:** Wikipedia'
    3. If you used Arxiv, you MUST end your answer with:
       '**Source:** Arxiv'
    4. If you used your own knowledge, do not include a source.
    """
    
    if pdf_processed:
        system_instruction += "\nA PDF is currently uploaded. Prioritize checking the PDF for answers."

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Searching..."):
            input_messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=user_input)
            ]
            
            events = agent.stream(
                {"messages": input_messages},
                config=config,
                stream_mode="values"
            )
            
            for event in events:
                if "messages" in event:
                    latest_msg = event["messages"][-1]
                    if latest_msg.type == "ai":
                        full_response = latest_msg.content
                        message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
