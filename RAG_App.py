import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Load environment variables
load_dotenv()

# Set up Groq API key
groq_api_key = st.secrets["GROQ_API_KEY"]  # Make sure this is set in .streamlit/secrets.toml

st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.title("Dynamic RAG with Groq, FAISS, and Llama3")

# Initialize session state
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - Document Upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        docs = []
        os.makedirs("temp_files", exist_ok=True)

        for uploaded_file in uploaded_files:
            temp_path = os.path.join("temp_files", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        if splits:
            try:
                embeddings = HuggingFaceEmbeddings()
                st.session_state.vector = FAISS.from_documents(splits, embeddings)
                st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"Failed to create vector store: {e}")
        else:
            st.warning("No content extracted from the uploaded PDFs.")
    else:
        st.info("Please upload at least one document.")

# Main Chat Section
st.header("Chat with your Documents")

# Set up the language model
llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# Display past messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User chat input
if prompt_input := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector is not None:
        with st.chat_message("user"):
            st.markdown(prompt_input)
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})

        with st.spinner("Thinking..."):
            try:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start = time.process_time()
                response = retrieval_chain.invoke({"input": prompt_input})
                response_time = time.process_time() - start

                answer = response.get("answer", "Sorry, I couldn't find an answer.")

            except Exception as e:
                answer = f"An error occurred: {str(e)}"
                response_time = 0.0

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.info(f"Response time: {response_time:.2f} seconds")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please process your documents before asking questions.")

