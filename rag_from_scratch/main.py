# main.py

import time
from rag_from_scratch.file_parser import parse_file_with_nltk
from rag_from_scratch.embedding_handler import get_embeddings, get_embedding
from rag_from_scratch.similarity_calculator import get_similarity_score
import ollama

def main():
    filename = 'War_and_Peace.txt'
    embedding_model = 'nomic-embed-text'
    
    content = parse_file_with_nltk(filename)
    
    start = time.perf_counter()
    content_embed = get_embeddings(content, filename, embedding_model)
    
    prompt = input("What do you want to know? -> ").strip()
    if not prompt:
        return False
    
    start = time.perf_counter()
    prompt_embed = get_embedding(embedding_model, prompt)
    
    similarity_scores = get_similarity_score(prompt_embed, content_embed)[:5]
    
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
    based on snippets of text provided in context. Answer only using the context provided,
    being as concise as possible. If you're unsure, just say that you don't know.
    Context:
    """
    
    response = ollama.chat(
        model='llama3.1',
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT + "\n".join(content[index] for score, index in similarity_scores)
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    
    print(response['message']['content'])
    return True

if __name__ == '__main__':
    while True:
        if not main():
            print("Exiting the program. Goodbye!")
            break
