# 📚 Multi-Mode RAG Bot  
## Made by Aelaf Eskindir
🔗 **Live Deployment:** [Click Here to Try the App 🚀](https://multi-mode-rag-bot-9hi4kdbx4c6pxbst6wcvxx.streamlit.app/)
🔗 **Github Repository:** [Click Here to get the source code 🚀](https://github.com/aelaf-git/Multi-Mode-RAG-Bot/)  

---

## ✨ Overview  
The **Multi-Mode RAG Bot** is a **Streamlit-powered AI assistant** that allows you to:  
- 📖 **Chat (Q&A)** with your uploaded PDF documents using **RAG (Retrieval-Augmented Generation)**.  
- 📝 **Summarize** entire documents with a **map-reduce summarization chain**.  

It integrates **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Groq LLMs** to deliver a smooth and interactive experience.  

---

## ⚙️ How It Works  

### 🔑 API Authorization  
- The app uses your **GROQ API Key**, securely stored in `secrets.toml`.  
- Keys are never exposed in the codebase.  

### 📂 Document Processing  
1. Upload a PDF file in the **sidebar**.  
2. The app will:  
   - Save it temporarily  
   - Extract the text using **PyPDFLoader**  
   - Split it into manageable chunks with **RecursiveCharacterTextSplitter**  

### 🧠 Knowledge Base Creation  
- Each document chunk is embedded using **HuggingFace MiniLM embeddings**.  
- **FAISS** builds a vector database for fast retrieval.  

### 💬 Q&A Mode  
- Queries are matched against the document using FAISS retriever.  
- A **cross-encoder reranker** refines the best results.  
- The selected context is passed into **Groq’s LLaMA-3 model** for precise answers.  

### 📝 Summarization Mode  
- A **map-reduce summarization chain** condenses the document into an easy-to-read summary.  

---

## 🖥️ User Interface  

### Sidebar  
1. 📤 Upload your PDF file  
2. ✅ Get confirmation once it’s loaded  

### Main Panel  
- Choose between **Q&A (Chat)** or **Summarization** mode.  

#### Q&A (Chat)  
- Ask questions in the chat box  
- Get **contextual answers** based only on the uploaded PDF  

#### Summarization  
- Click **"Generate Summary"** to get a concise overview of the document  

---

### Tech Stack

- Streamlit – Web app framework

- LangChain – AI orchestration

- Groq – LLaMA-3 inference

- FAISS – Vector search

- HuggingFace Transformers – Embeddings & reranker

---

### 📌 Example Use Cases

- 📚 Research papers → Summarize and Q&A

- 🏛️ Legal documents → Extract key points quickly

- 🧑‍🎓 Study material → Summarize chapters or test knowledge

- 📊 Reports → Fast insights

