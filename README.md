
# 📄 RAG-based Conversational PDF Assistant  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![LangChain](https://img.shields.io/badge/LangChain-Framework-orange)  
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)  
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)  

---

## **📌 About**  
This project lets you **chat with your PDF documents using AI**. It uses a **Retrieval-Augmented Generation (RAG)** approach, where your document is processed and searched for relevant information to give accurate answers. The system is built with **LangChain** and uses **Google Gemini** as the Large Language Model (LLM) to understand your questions and respond based on the PDF content.

---

## **✨ Features**  
✅ **Upload PDFs** – Add one or more PDF files to interact with.  
✅ **Text Extraction** – Extracts text from uploaded PDFs.  
✅ **Text Chunking** – Splits large text into smaller chunks for efficient search.  
✅ **Vector Store** – Uses **FAISS** for fast semantic similarity search.  
✅ **Conversational AI** – Ask questions, get accurate answers from your documents.  

---

## **🛠 Tech Stack**  
- **Python** – Core programming language  
- **Streamlit** – Web application framework  
- **PyPDF2** – PDF parsing and text extraction  
- **LangChain** – RAG pipeline and orchestration  
- **FAISS** – Vector database for similarity search  
- **Google Gemini** – LLM for intelligent responses  

---

## **⚙ How It Works**  
1. **Upload PDFs** via Streamlit UI.  
2. **Text Extraction & Chunking** for handling large documents.  
3. **Embeddings + FAISS Indexing** for semantic search.  
4. **Query Processing** – User asks a question.  
5. **RAG Pipeline** retrieves relevant chunks and generates answers with Gemini LLM.  

---


## **🔮 Future Enhancements**  
- ✅ Support for multiple file formats (Word, Excel).  
- ✅ Chat history and session management.  
 
 
