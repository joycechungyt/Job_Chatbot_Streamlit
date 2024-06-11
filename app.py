import streamlit as st
from streamlit_chat import message
import tempfile   # temporary file
from langchain_community.document_loaders.csv_loader import CSVLoader  # using CSV loaders
from langchain.embeddings import HuggingFaceEmbeddings # import hf embedding
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss' # Set the path of our generated embeddings

# Read the CSV files and combine job information
csv_path = "indeed_jobs.csv"
linkedin_csv_path = "linkedinjobs.csv"
linkedin_job_details_csv_path = "linkedin_job_details.csv"

loader = CSVLoader(file_path=[csv_path, linkedin_csv_path, linkedin_job_details_csv_path])
data = loader.load()

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create a FAISS vector store
vectorstore = FAISS.from_documents(data, embeddings)

# Initialize the LLM
llm = CTransformers(model_name="google/gemma-2b-it")

# Create the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Function to recommend a job based on user input
def recommend_job(user_input):
    result = qa({"question": user_input, "chat_history": st.session_state["history"]})
    st.session_state["history"].append((user_input, result["result"]))
    return result["result"]

# Streamlit app
st.title("Job Recommendation Chatbot")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("Enter your query or type 'recommend' to get a job recommendation:")

if user_input:
    if user_input.lower() == "recommend":
        recommendation = recommend_job("Please recommend a job based on my preferences.")
        st.write(recommendation)
    else:
        response = recommend_job(user_input)
        st.write(response)