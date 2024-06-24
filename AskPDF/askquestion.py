import numpy as np
from sentence_transformers import SentenceTransformer
import redis
from redis.commands.search.query import Query
from openai import OpenAI
from secrets import OPENAI_API_KEY 


OPENAI_API_KEY = "sk-proj-GmFiX5wa9Ym49IW33ht0T3BlbkFJQ6u5or7AXnvTvhAc8Isc"
def search_similar_chunks(query, top_k=5, redis_host='localhost', redis_port=6379):
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
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
        # print(doc)
        matching_chunks.append(doc.chunk) 
    context = "\n\n".join(matching_chunks)
    return context
    # return matching_chunks

def ask_openai(context, question):
    # Define parameters for the completion call
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
    print(chat_completion.choices[0].message.content)


query = "tell me about proximity graph"
context = search_similar_chunks(query)

ask_openai(context,query)
# print(context)
