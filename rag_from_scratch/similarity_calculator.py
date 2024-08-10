# similarity_calculator.py

import numpy as np
from numpy.linalg import norm

def get_similarity_score(query_embed, content_embed):
    query_norm = norm(query_embed)
    
    similarity_scores = [np.dot(query_embed, paragraph_embed) / (query_norm * norm(paragraph_embed)) for paragraph_embed in content_embed]
    sorted_scores = sorted(zip(similarity_scores, range(len(content_embed))), reverse=True)
    
    return sorted_scores
