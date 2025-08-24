import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

load_dotenv()
APP_PASSWORD = os.getenv("SMARTPDF_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided answers in a concise, shorter manner.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'

st.sidebar.title("🔐 Access & Model Settings")
user_password = st.sidebar.text_input("Enter Password", type="password")

if user_password == APP_PASSWORD:
    EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    LANGUAGE_MODEL = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

    st.title("📑 SmartPDF Expert")
    st.markdown("### Intelligent Expert Document Assistant")
    st.markdown("---")

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
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | LANGUAGE_MODEL
        result = chain.invoke({
            "user_query": user_query,
            "document_context": context_text
        })
        return result.content if hasattr(result, "content") else str(result)

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

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="text-align:center;"><p style="font-size:16px; color:blue;"><strong>Developed By Ganesh Komati</strong></p></div>',
    unsafe_allow_html=True
)