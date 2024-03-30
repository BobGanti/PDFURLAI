import os
import openai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv(".env.local")
openai.api_key = os.getenv('OPENAI_API_KEY')
INSTRUCTIONS = os.getenv('INSTRUCTIONS', 'Please follow the instructions.') 
ASSISTANT_PROFILE = os.getenv('PA_PROFILE', 'Default Assistant Profile')


# DocumentChunk class to wrap text chunks
class DocumentChunk:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Utility Functions
def chunk_text(text, chunk_size=2500):
    """Splits text into chunks without cutting words in half."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    while text:
        split_point = text.rfind(' ', 0, chunk_size) + 1
        if not split_point:  # No spaces found, hard split at chunk_size
            split_point = chunk_size
        chunks.append(text[:split_point].strip())
        text = text[split_point:]
    return chunks

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += (page.extract_text() or '') + '\n'
    return text

# Core Logic Functions
def create_vector_store_from_pdf(pdf_file):
    """Creates a vector store from an uploaded PDF file."""
    text = extract_text_from_pdf(pdf_file)
    doc_chunks = chunk_text(text)
    document_objects = [DocumentChunk(chunk) for chunk in doc_chunks]
    vector_store = Chroma.from_documents(document_objects, OpenAIEmbeddings())
    return vector_store

def get_response(user_query):
    """Generates a response for the user query using the conversation chain."""
    if 'vector_stores' in st.session_state:
        responses = []
        for vector_store in st.session_state['vector_stores']:
            retriever_chain = get_context_retriever_chain(vector_store)
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            response = conversation_rag_chain.invoke({
                        'chat_history': st.session_state.get('chat_history', []),
                        'input': user_query
                    })
            responses.append(response['answer'])
        return ' '.join(responses)
    return 'Unable to process the query without a valid vector store.'

# Streamlit UI and Main Workflow Functions
def setup_streamlit_ui():
    """Sets up Streamlit UI components."""
    st.set_page_config(page_title="PA", page_icon="📄")
    st.title("Personal Assistant")
    with st.sidebar:
        st.header("Menu")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        if uploaded_files:
            st.session_state['pdf_files'] = uploaded_files

def load_sys_pdfs():
    """Load PDF files from the default directory."""
    SYS_PDFs_DIR = os.path.join(os.path.dirname(__file__), 'system_files')
    SYS_PDFs = []
    for filename in os.listdir(SYS_PDFs_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(SYS_PDFs_DIR, filename)
            SYS_PDFs.append(file_path)
    return SYS_PDFs

def process_pdfs(pdf_files):
    """Processes uploaded PDF files."""
    vectorstores = [create_vector_store_from_pdf(pdf) for pdf in pdf_files]
    st.session_state['vector_stores'] = vectorstores

def display_conversation():
    """Displays conversation history and input box for new queries."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [] #  Initialising the chat history

    user_query = st.chat_input("Say something...", key="user_query")
    if user_query:
        response = get_response(user_query)
        st.session_state['chat_history'].extend([
            HumanMessage(content=user_query),
            AIMessage(content=response)
        ])

    for message in st.session_state['chat_history']:
        if isinstance(message, AIMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        else:
            with st.chat_message("AI"):
                st.write(message.content)

def get_context_retriever_chain(vector_store):
    """Creates a retriever chain for context retrieval."""
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", INSTRUCTIONS)
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    """Creates a chain for conversational RAG processing."""
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{ASSISTANT_PROFILE}:\n\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def main():

    setup_streamlit_ui()

    pdf_files = st.session_state.get('pdf_files', [])
    if pdf_files and st.sidebar.button("Process PDFs"):
        with st.sidebar:
            with st.spinner("Processing..."):
                process_pdfs(pdf_files)
                st.success("Processing complete.")
    else:
        pdf_files = load_sys_pdfs()
        process_pdfs(pdf_files) 

    display_conversation()

if __name__ == "__main__":
    main()
