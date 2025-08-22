import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from google import genai  # Gemini client
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Initialize Gemini client
gemini_client = genai.Client()

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(tools, ques):
    # Retrieve relevant context from PDF
    prompt_context = tools.run(ques)

    # Create the prompt for Gemini
    prompt = f"""
You are a helpful assistant. Answer the question as detailed as possible from the provided context.
If the answer is not in the provided context, just say, "Answer is not available in the context." 
Don't provide the wrong answer.

Context:
{prompt_context}

Question:
{ques}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"text": prompt}]
    )

    st.write("Reply:", response.text)

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(
        retriever,
        "pdf_extractor",
        "This tool is to give answers to queries from the PDF."
    )
    get_conversational_chain(retrieval_chain, user_question)

def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF using Gemini 2.5")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
