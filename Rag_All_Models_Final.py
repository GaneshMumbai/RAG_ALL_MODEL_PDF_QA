import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
APP_PASSWORD = os.getenv("SMARTPDF_PASSWORD")

# Prompt template
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided answers in a concise, shorter manner.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'

# Sidebar Configuration
st.sidebar.title("🔐 Access & Model Settings")

# Password input
user_password = st.sidebar.text_input("Enter Password", type="password")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select AI Model",
    options=["llama3:8b", "deepseek-r1:1.5b", "mistral:latest"],
    index=0,
    help="Choose the model for embeddings and responses"
)

# Proceed only if password is correct
if user_password == APP_PASSWORD:
    # Initialize model objects
    EMBEDDING_MODEL = OllamaEmbeddings(model=model_choice)
    LANGUAGE_MODEL = OllamaLLM(model=model_choice)
    DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

    # Main UI
    st.title("📑 SmartPDF Expert")
    st.markdown("### Intelligent Expert Document Assistant")
    st.markdown("---")

    # Ensure storage directory exists
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

    def save_uploaded_file(uploaded_file):
        file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return file_path

    def load_pdf_documents(file_path):
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()

    def chunk_documents(raw_documents):
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        return text_processor.split_documents(raw_documents)

    def index_documents(document_chunks):
        DOCUMENT_VECTOR_DB.add_documents(document_chunks)

    def find_related_documents(query):
        return DOCUMENT_VECTOR_DB.similarity_search(query)

    def generate_answer(user_query, context_documents):
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | LANGUAGE_MODEL
        return response_chain.invoke({
            "user_query": user_query,
            "document_context": context_text
        })

    # File Upload Section
    uploaded_pdf = st.file_uploader(
        "Upload Your Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )

    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)

        st.success("✅ Document processed successfully! Ask your questions below.")

        user_input = st.chat_input("Enter your question about the document...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Analyzing document..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)

            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)

else:
    st.warning("🔒 Please enter the correct password to access the assistant.")

# Sidebar Footer
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="text-align:center;"><p style="font-size:16px; color:blue;"><strong>Developed By Ganesh Komati</strong></p></div>',
    unsafe_allow_html=True
)
