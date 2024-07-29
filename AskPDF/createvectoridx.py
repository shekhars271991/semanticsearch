import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType



def create_redis_index(index_name="idxpdf", chunk_prefix="chunk_"):
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
    idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=[chunk_prefix])

    # Drop existing index if exists
    try:
        r.ft(index_name).dropindex()
    except:
        pass
    r.ft(index_name).create_index(schema, definition=idx_def)
