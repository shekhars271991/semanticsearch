from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Get the dimensionality of the embeddings
vector_dimensions = model.get_sentence_embedding_dimension()

print("Embedding dimensions:", vector_dimensions)