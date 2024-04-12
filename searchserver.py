from flask import Flask, request, jsonify
from redis.commands.search.query import Query
import numpy as np
import redis
from embeddings import generate_embeddings

app = Flask(__name__)
r = redis.Redis(host="localhost", port=6739)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        sampletext = data.get('text')
        if not sampletext:
            return jsonify({'error': 'Text field is required'}), 400

        embeddings = generate_embeddings(sampletext).tolist()
        vec = np.array(embeddings, dtype=np.float32).tobytes()

        # Check if 'genre' is provided in the query parameters
        genre = request.args.get('genre')
        if genre:
            # If genre is provided, construct the query with genre filter
            q = Query(f'(@genre:{genre})=>[KNN 3 @vector $query_vec AS vector_score]')\
                .sort_by('vector_score')\
                .return_fields('vector_score', 'content')\
                .dialect(2)
        else:
            # If genre is not provided, construct the query without genre filter
            q = Query('*=>[KNN 3 @vector $query_vec AS vector_score]')\
                .sort_by('vector_score')\
                .return_fields('vector_score', 'content')\
                .dialect(2)

        params = {"query_vec": vec}

        results = r.ft('idxadv').search(q, query_params=params)
        response = [{'distance': round(float(doc['vector_score']), 3), 'key': doc['id']} for doc in results.docs]
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
