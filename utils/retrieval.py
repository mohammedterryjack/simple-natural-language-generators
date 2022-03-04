from scipy.spatial.distance import cosine
from numpy import ndarray

def measure_similarity(embedding_a:ndarray, embedding_b:ndarray) -> float:
    return 1.0-cosine(embedding_a, embedding_b)