import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

# --- PAGE CONFIG ---
st.set_page_config(page_title="🌾 Naive RAG Chatbot", page_icon="🌾")

# --- TITLE / INTRO ---
st.title("🌾 Naive RAG Chatbot")
st.write("👋 Hello! Upload a document (PDF, Word, or TXT) and ask questions about it.")

# --- API KEY (for Groq) ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# --- EMBEDDINGS ---
embeddings = HuggingFaceEmbeddings()

# --- PROMPT TEMPLATE ---
prompt_template = """
Use the following pieces of context to answer the question.
If you don’t know the answer from the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- SESSION STATE INIT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- DOCUMENT UPLOAD ---
uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, TXT):",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    file_type = uploaded_file.type
    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(uploaded_file)
    else:
        loader = TextLoader(uploaded_file)
    
    docs = loader.load()
    
    # Optionally chunk documents
    # Here we just keep full content, or implement chunking if needed
    # For simplicity, we'll keep as one document
    st.session_state.retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    st.success(f"Document '{uploaded_file.name}' loaded successfully! You can now ask questions.")

# --- USER INPUT FORM ---
with st.form("chat_form", clear_on_submit=True):
    if st.session_state.retriever is None:
        st.text_input("💬 Ask a question:", disabled=True, placeholder="Please upload a document first.")
        submitted = False
    else:
        user_input = st.text_input("💬 Ask a question:")
        submitted = st.form_submit_button("Send")
    
        if submitted and user_input.strip() != "":
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            with st.spinner("Thinking..."):
                answer = qa_chain.run(user_input)
            
            # Insert user input at the top
            st.session_state.chat_history.insert(0, ("User", user_input))
            
            # First-time greeting
            if not st.session_state.greeted:
                greeting = "👋 Hello! I’m your Crop Advisor bot. How can I help you today?"
                st.session_state.chat_history.insert(0, ("Bot", greeting))
                st.session_state.greeted = True
            
            # Insert bot answer at the top
            st.session_state.chat_history.insert(0, ("Bot", answer))

# --- DISPLAY CHAT HISTORY (newest at top) ---
for speaker, message in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"<div style='background-color:#D1E7DD;padding:8px;border-radius:8px;margin-bottom:5px'><b>User:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#F8D7DA;padding:8px;border-radius:8px;margin-bottom:5px'><b>Bot:</b> {message}</div>", unsafe_allow_html=True)
