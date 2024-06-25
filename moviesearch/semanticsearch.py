from redis.commands.search.query import Query
import numpy as np
import redis
from embeddings import generate_embeddings
sampletext = "a travel story going to differnt locations"
r = redis.Redis(host="localhost", port=6379)

embeddings = generate_embeddings(sampletext).tolist()

vec = np.array(embeddings, dtype=np.float32).tobytes()
# q = Query('*=>[KNN 3 @vector $query_vec AS vector_score]')\
#     .sort_by('vector_score')\
#     .return_fields('vector_score', 'content')\
#     .dialect(2)    
# params = {"query_vec": vec}

q = Query('(@genre:Comedy)=>[KNN 3 @vector $query_vec AS vector_score]')\
    .sort_by('vector_score')\
    .return_fields('vector_score', 'content')\
    .dialect(2)    
params = {"query_vec": vec, "genre": "Comedy"}



results = r.ft('idxadv').search(q, query_params=params)

for doc in results.docs:
    print(f"distance:{round(float(doc['vector_score']),3)} key: {doc['id']}\n")