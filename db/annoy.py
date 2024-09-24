# Annoy index functions
from annoy import AnnoyIndex
import numpy as np
from db.embeddings import EmbeddingDB


def build_annoy_index(conn: EmbeddingDB, vector_size=4096, n_trees=10):
    
    total_vectors = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    
    annoy_index = AnnoyIndex(vector_size, 'angular')
    
    for i, (id, embedding_blob) in enumerate(conn.execute("SELECT id, embedding FROM embeddings").fetchall()):
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        if len(embedding) != vector_size:
            print(f"Warning: Embedding size mismatch. Expected {vector_size}, got {len(embedding)}. Skipping this vector.")
            continue
        annoy_index.add_item(id - 1, embedding)
    
    print("Building index...")
    annoy_index.build(n_trees)
    annoy_index.save('embeddings.ann')
    print("Index built and saved")

def find_similar(conn: EmbeddingDB, query_embedding, top_k=5):
    annoy_index = AnnoyIndex(4096, 'angular')
    annoy_index.load('embeddings.ann')
    
    similar_ids, distances = annoy_index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
    
    results = []
    for id, distance in zip(similar_ids, distances):
        text, is_question = conn.execute("SELECT text, is_question FROM embeddings WHERE id = ?", (id + 1,)).fetchone()
        similarity = 1 - distance
        results.append((id + 1, text, similarity, bool(is_question)))
    
    return results