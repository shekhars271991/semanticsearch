from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def generate_embeddings(text: str) -> list[float]:
    embeddings = model.encode(text)
    return embeddings
