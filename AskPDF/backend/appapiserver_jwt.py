from flask import Flask, request, jsonify
from flask_cors import CORS
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
import json
from auth import token_required, encode_jwt
import logging




app = Flask(__name__)
CORS(app)
model = SentenceTransformer('all-MiniLM-L6-v2')
UPLOAD_DIRECTORY = './AskPDF/uploadedFiles'
redis_client = redis.Redis(host="localhost", port=6379, db=0)
INDEX_NAME = "idxpdf"


@app.before_request
def log_request_info():
    logging.info('Headers: %s', request.headers)
    logging.info('Body: %s', request.get_data())
    
# Define your routes here
@app.route('/ask', methods=['POST'])
@token_required
def ask_question():
    data = request.get_json()
    query = data['query']
    role = data['role']

    context = search_similar_chunks(query, role)
    answer = ask_openai(context, query)

    return jsonify({'answer': answer})

@app.route('/upload', methods=['POST'])
@token_required
def upload_file():
    roles_string = request.form.get('roles')
    roles = roles_string.split(',')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        unique_filename = get_unique_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
        file.save(file_path)

        try:
            doc_name = os.path.splitext(unique_filename)[0]
            upload_time = datetime.now().isoformat()

            text = extract_text_from_pdf(file_path)
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)
            store_file_metadata(doc_name, file.filename, upload_time, roles)
            store_in_redis(doc_name, chunks, embeddings, roles)

            return jsonify({'message': 'Document processed and embeddings stored in Redis'}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to process document: {str(e)}'}), 500

@app.route('/documents', methods=['GET'])
@token_required
def get_uploaded_documents():
    documents = list_uploaded_documents()
    return jsonify({'documents': documents})

@app.route('/delete', methods=['DELETE'])
@token_required
def delete_document():
    data = request.get_json()
    doc_name = data.get('doc_name')

    if not doc_name:
        return jsonify({'error': 'Missing document name'}), 400

    try:
        file_path = os.path.join(UPLOAD_DIRECTORY, f"{doc_name}.pdf")
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            return jsonify({'error': 'File not found in filesystem'}), 404

        metadata_key = f"file_{doc_name}_metadata"
        redis_client.delete(metadata_key)

        chunk_keys = redis_client.keys(f"chunk_{doc_name}_*")
        for key in chunk_keys:
            redis_client.delete(key)

        return jsonify({'message': f'Document {doc_name} deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to delete document: {str(e)}'}), 500

@app.route('/login', methods=['POST'])
def login():
    auth = request.get_json()
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Username and password required!'}), 400
    
    # Here you should verify username and password with your user database
    # This example assumes authentication is successful
    token = encode_jwt({'username': auth.get('username')})
    return jsonify({'token': token})

def search_similar_chunks(query, role):
    query_embedding = model.encode(query)
    vector = np.array(query_embedding, dtype=np.float32).tobytes()

    q = Query(f'(@roles:{role} | public)=>[KNN 3 @vector $query_vec AS vector_score]')\
                .sort_by('vector_score')\
                .return_fields('vector_score', 'chunk')\
                .dialect(3)

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

def store_file_metadata(doc_name, original_filename, upload_time, roles):
    metadata_key = f"file_{doc_name}_metadata"
    metadata = {
        "uploaded_time": upload_time,
        "original_filename": original_filename,
        "roles": roles
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

def get_unique_filename(filename):
    original_filename = filename
    filename_prefix = original_filename[:3] if len(original_filename) >= 3 else original_filename
    filename, file_extension = os.path.splitext(original_filename)
    unique_filename = f"{filename_prefix}_{str(uuid.uuid4())[:8]}{file_extension}"
    return unique_filename

def store_in_redis(doc_name, chunks, embeddings, roles):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        key = f"chunk_{doc_name}_{i}"
        value = {
            "chunk": chunk,
            "embedding": embedding.tolist(),
            "roles": roles
        }
        redis_client.json().set(key, '.', value)

def list_uploaded_documents():
    document_keys = redis_client.keys('file_*')
    documents = []
    for key in document_keys:
        if key.decode('utf-8').endswith('_metadata'):
            doc_name = key.decode('utf-8').split('_metadata')[0].split('file_')[1]
            metadata = redis_client.json().get(key)
            documents.append({
                "doc_name": doc_name,
                "metadata": metadata
            })
    return documents

def create_vector_index():
    schema = [
        TextField("$.chunk", as_name='chunk'),
        TextField("$.roles", as_name='roles'),
        VectorField('$.embedding', "HNSW", {
            "TYPE": 'FLOAT32',
            "DIM": 384,
            "DISTANCE_METRIC": "COSINE"
        }, as_name='vector')
    ]

    idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=['chunk_'])

    try:
        # redis_client.ft(INDEX_NAME).dropindex()
        redis_client.ft(INDEX_NAME).create_index(schema, definition=idx_def)

    except:
        pass
    # redis_client.ft(INDEX_NAME).create_index(schema, definition=idx_def)

if __name__ == '__main__':
    create_vector_index()
    app.run(debug=True, use_reloader=False)
