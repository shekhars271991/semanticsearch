import streamlit as st
import requests
import json

# Configuration
FLASK_API_URL = "http://localhost:5000"

# Streamlit app
st.title("PDF Question Answering App")

# Function to list uploaded documents
def list_uploaded_documents():
    response = requests.get(f"{FLASK_API_URL}/documents")
    if response.status_code == 200:
        documents = response.json().get('documents')
        return documents
    else:
        st.error("Failed to list documents.")
        return []

# Upload PDF
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
roles = st.text_input("Enter roles (comma separated)", "")

if st.button("Upload"):
    if uploaded_file and roles:
        files = {'file': uploaded_file}
        data = {'roles': roles}
        response = requests.post(f"{FLASK_API_URL}/upload", files=files, data=data)
        if response.status_code == 200:
            st.success("PDF uploaded and processed successfully.")
            st.experimental_rerun()
        else:
            st.error(f"Failed to upload PDF: {response.json().get('error')}")
    else:
        st.error("Please upload a file and enter roles.")

# Ask a question
st.header("Ask a Question")
question = st.text_area("Enter your question")
role = st.text_input("Enter the role to search within", "")

if st.button("Ask"):
    if question and role:
        data = {'query': question, 'role': role}
        response = requests.post(f"{FLASK_API_URL}/ask", json=data)
        if response.status_code == 200:
            answer = response.json().get('answer')
            st.success(f"Answer: {answer}")
        else:
            st.error(f"Failed to get answer: {response.json().get('error')}")
    else:
        st.error("Please enter a question and a role.")

# List uploaded documents
st.header("Uploaded Documents")
documents = list_uploaded_documents()

if documents:
    for doc in documents:
        col1, col2, col3, col4 = st.columns([3, 3, 2, 1])
        with col1:
            st.write(f"Document Name: {doc['doc_name']}")
        with col2:
            st.write(f"Uploaded Time: {doc['metadata']['uploaded_time']}")
        with col3:
            st.write(f"Original Filename: {doc['metadata']['original_filename']}")
        with col4:
            if st.button("❌", key=doc['doc_name']):
                data = {'doc_name': doc['doc_name']}
                response = requests.delete(f"{FLASK_API_URL}/delete", json=data)
                if response.status_code == 200:
                    st.success(f"Document {doc['doc_name']} deleted successfully.")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to delete document: {response.json().get('error')}")
        st.write("---")
else:
    st.write("No documents uploaded yet.")
