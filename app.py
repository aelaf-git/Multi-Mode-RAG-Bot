import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- Caching Functions to Avoid Re-running Expensive Processes ---
"authorization": st.secrets["GROQ_API_KEY"]

# Cache the function that loads and splits the document
@st.cache_data
def load_and_split_document(uploaded_file):
    """Loads a PDF from an uploaded file, saves it temporarily, and splits it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info("Loading and processing the PDF...")
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    doc_splits = text_splitter.split_documents(documents)
    
    # Clean up the temporary file
    os.remove(tmp_file_path)
    
    return doc_splits

# Cache the function that creates the vector store for Q&A
@st.cache_resource
def create_vector_store(_doc_splits): # Underscore to signal it's a cached resource
    """Creates a FAISS vector store from document splits."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_documents(_doc_splits, embedding_model)
    st.success("Vector store for Q&A is ready!")
    return vector_store

# Cache the RAG chain creation
@st.cache_resource
def create_rag_chain(_vector_store):
    """Creates the retriever, reranker, and the full RAG chain for Q&A."""
    st.info("Setting up the RAG chain...")
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="llama3-8b-8192",
        temperature=0
    )
    retriever = _vector_store.as_retriever()
    cross_encoder_model = HuggingFaceCrossEncoder(
        model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        model_kwargs={'device': 'cpu'}
    )
    reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=2)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )
    prompt_template = """
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise and based ONLY on the provided context.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain

# Cache the Summarization chain creation
@st.cache_resource
def create_summarization_chain():
    """Creates a map-reduce summarization chain."""
    st.info("Setting up the Summarization chain...")
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="llama3-8b-8192",
        temperature=0.2 
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    st.success("Summarization chain is ready!")
    return chain

# --- Streamlit App Interface ---

st.set_page_config(page_title="Multi-Mode RAG Bot", page_icon="📚")
st.title("📚 Multi-Mode RAG Bot")
st.write("Upload a PDF, then choose to either chat with it (Q&A) or get a summary.")

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

# Main panel
if not uploaded_file:
    st.info("Please upload a PDF document in the sidebar to begin.")
else:
    # Process the document
    doc_splits = load_and_split_document(uploaded_file)
    
    # Mode selection
    mode = st.radio(
        "2. Choose your mode:",
        ("Q&A (Chat)", "Summarization"),
        horizontal=True
    )

    if mode == "Q&A (Chat)":
        # Create the vector store and RAG chain for Q&A
        vector_store = create_vector_store(doc_splits)
        rag_chain = create_rag_chain(vector_store)
        
        st.header("Q&A with Your Document")

        # Initialize chat history
        if "messages" not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your document."}]
            st.session_state['file_name'] = uploaded_file.name # Track current file

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    elif mode == "Summarization":
        # Create the summarization chain
        summarize_chain = create_summarization_chain()
        
        st.header("Summarize Your Document")
        if st.button("Generate Summary"):
            with st.spinner("Generating summary... This can take some time for large documents."):
                # The `invoke` method for this chain expects a dictionary with 'input_documents'
                summary = summarize_chain.invoke({"input_documents": doc_splits})
                st.subheader("Summary")

                st.write(summary['output_text'])


