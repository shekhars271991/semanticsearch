# you can download data from
# https://www.kaggle.com/datasets/bourdier/all-tv-series-details-dataset?select=tvs.json

import json
from redis import Redis
from redis.exceptions import RedisError
from sentence_transformers import SentenceTransformer
from embeddings import generate_embeddings
skip=0
uploaded=0
UPLOAD_LIMIT = 100000

try:
    # Connect to Redis
    redis_client = Redis(host='localhost', port=6379, db=0)

    # Load JSON data from file
    with open('data/tvs.json', 'r') as file:
        json_data = json.load(file)

    # Iterate over each JSON object
    for data in json_data:
        # Check if the JSON object has a non-empty 'overview' attribute
        if(uploaded == UPLOAD_LIMIT):
            break
        overview = data.get('overview', '')
        if overview:
            # Generate embeddings for the overview
            overview_embeddings = generate_embeddings(overview)
            
            # Add the embeddings to the JSON object
            data['overview_embeddings'] = overview_embeddings.tolist()

            # Upload JSON object to Redis
            key = "movie:"+data.get('name', '')  # Use the 'name' attribute as the key
            if key:
                redis_client.json().set(key, '.', data)
                print(f"Uploaded JSON with key '{key}' to Redis. ("+str(uploaded)+")")
                uploaded = uploaded+1
            else:
                print("Skipping JSON object without a 'name' attribute.")
                skip = skip +1
        else:
            print("Skipping JSON object without an 'overview' attribute or with an empty overview.")

except FileNotFoundError:
    print("Error: File not found.")
except json.JSONDecodeError:
    print("Error: Invalid JSON format in file.")
except RedisError as e:
    print(f"Redis Error: {e}")
finally:
    # Close the Redis connection
    print( "Skipped Count", str(skip))
    print( "uploaded Count", str(uploaded))
    if 'redis_client' in locals():
        redis_client.close()
