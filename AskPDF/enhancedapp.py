# ! pip3 install pymupdf sentence-transformers redis streamlit

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import redis
import json
import streamlit as st
from openai import OpenAI
from redis.commands.search.query import Query
import numpy as np
from my_secrets import OPENAI_API_KEY 


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    return embeddings

def store_in_redis(chunks, embeddings, redis_client):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        key = f"pdf_v1_chunk_{i}"
        value = {
            "chunk": chunk,
            "embedding": embedding.tolist()  # Convert to list for JSON serialization
        }
        redis_client.json().set(key,'.',value)

def process_pdf(pdf_path, redis_host='localhost', redis_port=6379):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks, model)
    
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
    store_in_redis(chunks, embeddings, redis_client)
    st.success("PDF processed and data stored in Redis.")

def search_similar_chunks(model, query, top_k=5, redis_host='localhost', redis_port=6379):
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
   
    
    query_embedding = model.encode(query)
    vector = np.array(query_embedding, dtype=np.float32).tobytes()
    q = Query('*=>[KNN 3 @vector $query_vec AS vector_score]')\
    .sort_by('vector_score')\
    .return_fields('vector_score', 'chunk')\
    .dialect(2)    
    params = {"query_vec": vector}

    results = redis_client.ft('idxpdf').search(q, query_params=params)

    matching_chunks = []
    for doc in results.docs:
        matching_chunks.append(doc.chunk) 
    context = "\n\n".join(matching_chunks)
    return context

def ask_openai(context, question):
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = template.format(context=context, question=question)
    client = OpenAI(api_key=OPENAI_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


# Streamlit UI
st.title("Upload PDF and ask questions")
model = SentenceTransformer('all-MiniLM-L6-v2')
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_path = f"/tmp/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        process_pdf(pdf_path)
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    context = search_similar_chunks(model, query)
    answer = ask_openai(context, query)
    st.subheader("Answer:")
    st.write(answer)


