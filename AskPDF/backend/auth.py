import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app as app

SECRET_KEY = 'your_secret_key'  # Replace with your actual secret key
ALGORITHM = 'HS256'
EXPIRATION_TIME = 3600  # Token expiration time in seconds

def encode_jwt(payload):
    expiration = datetime.utcnow() + timedelta(seconds=EXPIRATION_TIME)
    payload['exp'] = expiration
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_jwt(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Token has expired")
    except jwt.InvalidTokenError:
        raise jwt.InvalidTokenError("Invalid token")

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]  # Bearer token
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = decode_jwt(token)
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function
