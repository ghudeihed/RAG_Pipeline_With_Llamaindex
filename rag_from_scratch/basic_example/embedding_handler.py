# embedding_handler.py

import os
import json
import ollama

def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs('embeddings')
    
    try:
        with open(f'embeddings/{filename}.json', 'w') as f:
            json.dump(embeddings, f)
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def load_embeddings(filename):
    filepath = f'embeddings/{filename}.json'
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return False

def get_embedding(modelname, chunk):
    return ollama.embeddings(model=modelname, prompt=chunk)['embedding']

def get_embeddings(chunks, filename, modelname = 'nomic-embed-text'):
    embeddings = load_embeddings(filename)
    
    if not embeddings:
        embeddings = [get_embedding(modelname, chunk) for chunk in chunks]
        save_embeddings(filename, embeddings)
    
    return embeddings
