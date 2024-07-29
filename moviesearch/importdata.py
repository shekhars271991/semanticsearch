import json
from redis import Redis
from redis.exceptions import RedisError

try:
    # Connect to Redis
    redis_client = Redis(host='localhost', port=6379, db=0)

    # Load JSON data from file
    with open('../data/tvs.json', 'r') as file:
        json_data = json.load(file)

    # Iterate over each JSON object
    for data in json_data:
        # Use the 'name' attribute as the key
        key = data.get('name')
        if key:
            # Upload JSON object to Redis
            redis_client.json().set(key, '.', data)
            print(f"Uploaded JSON with key '{key}' to Redis.")
        else:
            print("Skipping JSON object without a 'name' attribute.")

except FileNotFoundError:
    print("Error: File not found.")
except json.JSONDecodeError:
    print("Error: Invalid JSON format in file.")
except RedisError as e:
    print(f"Redis Error: {e}")
finally:
    # Close the Redis connection
    if 'redis_client' in locals():
        redis_client.close()
