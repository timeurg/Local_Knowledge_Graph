
import re
import json
from sklearn.metrics.pairwise import cosine_similarity

def extract_json(text):
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()
    json_objects = re.findall(r'\{[^{}]*\}', text)
    
    if json_objects:
        try:
            return json.loads(json_objects[-1])
        except json.JSONDecodeError:
            pass
    
    return {
        "title": "Parsing Error",
        "content": text,
        "next_action": "continue"
    }

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def calculate_top_similarities(embeddings, current_step, top_k=2):
    similarities = []
    for i in range(min(current_step, len(embeddings))):
        if i < len(embeddings) and current_step < len(embeddings):
            similarity = float(calculate_similarity(embeddings[current_step], embeddings[i]))
            similarities.append((i, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]