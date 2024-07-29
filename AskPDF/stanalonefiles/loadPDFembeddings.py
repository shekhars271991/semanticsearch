# ! pip3 install pymupdf sentence-transformers redis


import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import redis
import json

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
        key = f"chunk_{i}"
        value = {
            "chunk": chunk,
            "embedding": embedding.tolist()  # Convert to list for JSON serialization
        }
        redis_client.json().set(key,'.',value)

def process_pdf(pdf_path, redis_host='localhost', redis_port=6379):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = get_embeddings(chunks, model)
    
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
    store_in_redis(chunks, embeddings, redis_client)
    print("PDF processing and storing in Redis completed.")

# Example usage
pdf_path = "/Users/shekharsuman/sampleredisvss/AskPDF/HSNWPaper.pdf"
process_pdf(pdf_path)
