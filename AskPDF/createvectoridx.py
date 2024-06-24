import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis
r = redis.Redis(host="localhost", port=6379)

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
    r.ft('idxpdf').dropindex()
except:
    pass
r.ft('idxpdf').create_index(schema, definition=idx_def)
