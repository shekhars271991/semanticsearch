import numpy as np
from sentence_transformers import SentenceTransformer
import redis
from redis.commands.search.query import Query
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import openai

OPENAI_API_KEY = "sk-proj-GmFiX5wa9Ym49IW33ht0T3BlbkFJQ6u5or7AXnvTvhAc8Isc"
openai.api_key = OPENAI_API_KEY
def search_similar_chunks(query, top_k=5, redis_host='localhost', redis_port=6379):
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = model.encode(query)
    
    # Perform the search
    # search_query = f'*=>[KNN {top_k} @embedding $vec AS score]'
    vector = np.array(query_embedding, dtype=np.float32).tobytes()
   
    

    q = Query('*=>[KNN 3 @vector $query_vec AS vector_score]')\
    .sort_by('vector_score')\
    .return_fields('vector_score', 'chunk')\
    .dialect(2)    
    params = {"query_vec": vector}

    results = redis_client.ft('idxpdf').search(q, query_params=params)



# results = r.ft('idxadv').search(q, query_params=params)
#     results = redis_client.ft().search(
#         search_query,
#         query_params,
#         return_fields=["chunk", "score"],
#         dialect=2
#     )

    matching_chunks = []
    for doc in results.docs:
        # print(doc)
        matching_chunks.append(doc.chunk) 
    context = "\n\n".join(matching_chunks)
    return context
    # return matching_chunks

# Example usage
        
# def answer_question_with_chunks(context, question, top_k=5):
#     # Perform vector similarity search to get relevant chunks
#     # relevant_chunks = search_similar_chunks(question, top_k=top_k)

#     model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")


#     template = """
#     Answer the question based on the context below. If you can't 
#     answer the question, reply "I don't know".

#     Context: {context}

#     Question: {question}
#     """

#     prompt = ChatPromptTemplate.from_template(template)
#     # prompt.format(context, question)
    
#     # Combine question and relevant chunks into contextcontext = "\n\n".join(relevant_chunks)
#     # combined_input = f"Question: {question}\n\nContext:\n{context}"

    

#     parser = StrOutputParser()

#     chain = prompt | model | parser
#     return chain.invoke({
#         "context": context,
#         "question": question
#     })
    
#     # Generate answer using the LLM
#     # inputs = tokenizer.encode(combined_input, return_tensors='pt')
#     # outputs = llm_model.generate(inputs, max_length=500, num_return_sequences=1)
#     # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # return answer
# def answer_question_with_chunks(context, question):
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = {
        "type": "langchain.prompts.ChatPromptTemplate",
        "template": template
    }

    parser = {
        "type": "langchain_core.output_parsers.StrOutputParser"
    }

    chain = {
        "inputs": {
            "context": context,
            "question": question
        },
        "steps": [prompt, model, parser]
    }

    # Invoke the chain
    result = model.invoke(chain)
    return result



# Set your OpenAI API key here


def ask_openai(context, question):
    # Define parameters for the completion call
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = template.format(context=context, question=question)


    response = openai.Completion.create(
        engine="davinci-codex",  # Use Davinci Codex engine for more programming context
        prompt=prompt,
        max_tokens=100  # Adjust the max tokens as needed
    )

    # Extract and return the answer from the completion
    answer = response.choices[0].text.strip()
    return answer



query = "tell me about proximity graph"
context = search_similar_chunks(query)

answer = ask_openai(context,query)
print(context)
