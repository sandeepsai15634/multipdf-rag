import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Multi PDF QA Bot", layout="wide")

st.title("📚 Multi PDF Question Answering Chatbot")
st.write("Upload multiple PDFs and ask questions from them.")


# ===============================
# Get API Key Securely
# ===============================
try:
    groq_api_key = os.environ["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 GROQ_API_KEY not found.")
    st.info("Local: set GROQ_API_KEY=your_key")
    st.info("Cloud: Add GROQ_API_KEY in Streamlit Secrets")
    st.stop()


# ===============================
# Load Models (Cached)
# ===============================
@st.cache_resource
def load_models(api_key):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    return embeddings, llm


embeddings, llm = load_models(groq_api_key)


# ===============================
# File Upload
# ===============================
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)


# ===============================
# Process PDFs
# ===============================
if st.button("Process PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs..."):

            all_docs = []

            for uploaded_file in uploaded_files:
                # Cloud-safe temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name

                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)

            # Split Documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(all_docs)

            # Create Vector Store
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # Prompt Template
            prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not found in the context, say:
"I cannot find the answer in the provided documents."

<context>
{context}
</context>

Question: {input}
""")

            document_chain = create_stuff_documents_chain(llm, prompt)

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            st.session_state.retrieval_chain = retrieval_chain

            st.success("✅ PDFs processed successfully!")


# ===============================
# Ask Question
# ===============================
query = st.text_input("Ask your question:")

if query:
    if "retrieval_chain" not in st.session_state:
        st.warning("⚠️ Please process PDFs first.")
    else:
        with st.spinner("Thinking..."):
            result = st.session_state.retrieval_chain.invoke(
                {"input": query}
            )

            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("View Source Chunks"):
                for doc in result["context"]:
                    st.write("Page:", doc.metadata.get("page", "Unknown"))
                    st.write(doc.page_content[:500])
                    st.write("---")