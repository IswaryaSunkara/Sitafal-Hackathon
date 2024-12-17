import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Load environment variables
load_dotenv()

# API Key verification
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    st.error("Error: ANTHROPIC_API_KEY is missing. Please set it in your environment.")
    st.stop()

# Initialize Spacy embeddings
embedding_model = SpacyEmbeddings(model_name="en_core_web_sm")

# Constants for UI Labels
APP_TITLE = "Chat with PDF Documents"
SIDE_MENU_TITLE = "Upload PDFs"
HEADER_TITLE = "Ask Questions from PDFs"
PROCESSING_MSG = "Extracting and Processing the PDFs..."
PROCESS_COMPLETE_MSG = "PDFs have been processed! Start asking questions now."
ERROR_READING_PDF = "Error occurred while reading PDF: "
NO_TEXT_WARNING = "No text found in the uploaded PDFs."
UPLOAD_WARNING = "Please upload at least one PDF file."
SUBMIT_BUTTON_LABEL = "Process PDFs"
QUESTION_PROMPT = "Enter your question about the uploaded PDFs here"

# Function to extract text from uploaded PDFs
def extract_pdf_text(pdf_files):
    combined_text = ""
    for uploaded_file in pdf_files:
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                combined_text += page.extract_text() or ""
        except Exception as e:
            st.error(f"{ERROR_READING_PDF} {uploaded_file.name} - {e}")
    return combined_text

# Function to split text into manageable chunks
def create_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Function to store text embeddings in a FAISS database
def create_vector_store(text_chunks):
    faiss_db = FAISS.from_texts(text_chunks, embedding=embedding_model)
    faiss_db.save_local("pdf_vector_store")

# Function to answer questions using the conversational chain
def handle_question(user_question):
    try:
        # Load the FAISS vector store
        faiss_db = FAISS.load_local("pdf_vector_store", embedding_model, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever()

        # Define the PDF retriever tool
        query_tool = Tool(
            name="PDF Query Tool",
            description="Retrieves relevant answers from the uploaded PDF data.",
            func=retriever.get_relevant_documents
        )

        # Initialize LLM and agent
        llm_model = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=API_KEY)
        custom_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that answers questions based on the given PDF context. "
                       "If the answer is not in the context, respond with: 'No relevant answer found in the provided PDFs.'"),
            ("human", "{input}")
        ])
        agent = create_tool_calling_agent(llm_model, [query_tool], custom_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[query_tool])

        # Get and display the response
        response = agent_executor.invoke({"input": user_question})
        st.subheader("Assistant's Response:")
        st.info(response['output'])

    except Exception as e:
        st.error(f"Error while processing the question: {e}")

# Streamlit App
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # Sidebar for PDF upload
    with st.sidebar:
        st.header(SIDE_MENU_TITLE)
        uploaded_pdfs = st.file_uploader("Upload PDF files here", accept_multiple_files=True, type=["pdf"])
        if st.button(SUBMIT_BUTTON_LABEL):
            if uploaded_pdfs:
                with st.spinner(PROCESSING_MSG):
                    extracted_text = extract_pdf_text(uploaded_pdfs)
                    if extracted_text:
                        text_chunks = create_text_chunks(extracted_text)
                        create_vector_store(text_chunks)
                        st.success(PROCESS_COMPLETE_MSG)
                    else:
                        st.warning(NO_TEXT_WARNING)
            else:
                st.warning(UPLOAD_WARNING)

    # User input for questions
    st.subheader(HEADER_TITLE)
    user_question = st.text_area(QUESTION_PROMPT)
    if st.button("Get Answer"):
        if user_question.strip():
            handle_question(user_question)
        else:
            st.warning("Please enter a valid question.")

if _name_ == "_main_":
    main()
