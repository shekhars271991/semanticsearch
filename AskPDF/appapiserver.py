# app.py

from flask import Flask, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
from redis.commands.search.query import Query
from openai import OpenAI
from my_secrets import OPENAI_API_KEY 
import fitz
import os
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import uuid
from datetime import datetime



app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
UPLOAD_DIRECTORY = './AskPDF/uploadedFiles'
redis_client = redis.Redis(host="localhost", port=6379, db=0)
INDEX_NAME = "idxpdf"

def search_similar_chunks(query):
   
    query_embedding = model.encode(query)
    vector = np.array(query_embedding, dtype=np.float32).tobytes()
    q = Query('*=>[KNN 3 @vector $query_vec AS vector_score]')\
    .sort_by('vector_score')\
    .return_fields('vector_score', 'chunk')\
    .dialect(2)    
    params = {"query_vec": vector}

    results = redis_client.ft(INDEX_NAME).search(q, query_params=params)

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
    return chat_completion.choices[0].message.content.strip()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data['query']

    context = search_similar_chunks(query)
    answer = ask_openai(context, query)

    return jsonify({'answer': answer})


# the following code is for uploading a PDF

# Function to store file metadata in Redis
def store_file_metadata(doc_name, original_filename, upload_time):
    metadata_key = f"file_{doc_name}_metadata"
    metadata = {
        "uploaded_time": upload_time,
        "original_filename": original_filename
    }
    redis_client.json().set(metadata_key, '.', metadata)


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

def get_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

def store_in_redis(doc_name, chunks, embeddings):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        key = f"{doc_name}_chunk_{i}"
        value = {
            "chunk": chunk,
            "embedding": embedding.tolist()  # Convert to list for JSON serialization
        }
        redis_client.json().set(key, '.', value)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Generate a unique filename
        original_filename = file.filename
        filename_prefix = original_filename[:3] if len(original_filename) >= 3 else original_filename
        filename, file_extension = os.path.splitext(original_filename)
        unique_filename = f"{filename_prefix}_{str(uuid.uuid4())[:8]}{file_extension}"

        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
        file.save(file_path)

        # Process the PDF
        try:
            doc_name = os.path.splitext(unique_filename)[0]
            upload_time = datetime.now().isoformat()

            text = extract_text_from_pdf(file_path)
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)
            store_file_metadata(doc_name, original_filename, upload_time)

            store_in_redis(doc_name, chunks, embeddings)
            
            return jsonify({'message': 'Document processed and embeddings stored in Redis'}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to process document: {str(e)}'}), 500



def create_vector_index():
    # Define the schema
    schema = [
        TextField("$.chunk", as_name='chunk'),
        VectorField('$.embedding', "HNSW", {
            "TYPE": 'FLOAT32',
            "DIM": 384,
            "DISTANCE_METRIC": "COSINE"
        }, as_name='vector')
    ]

# Define index definition
    idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=['chunk_'])

# Drop existing index if exists
    try:
        redis_client.ft(INDEX_NAME).dropindex()
    except:
        pass
    redis_client.ft(INDEX_NAME).create_index(schema, definition=idx_def)

if __name__ == '__main__':
    create_vector_index()
    app.run(debug=True)
