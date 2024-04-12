import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis
r = redis.Redis(host="localhost", port=6739)

# Define the schema
schema = [
    VectorField('$.overview_embeddings', "FLAT", {
        "TYPE": 'FLOAT32',
        "DIM": 384,
        "DISTANCE_METRIC": "COSINE"
    }, as_name='vector'),
    TextField('$.name', as_name='name'),
    TextField('$.genres[*].name', as_name='genre')
]

# Define index definition
idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=['movie:'])

# Drop existing index if exists
try:
    r.ft('idxadv').dropindex()
except:
    pass
r.ft('idxadv').create_index(schema, definition=idx_def)
