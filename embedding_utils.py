# calculate the cosine similarity between database face and cctv face
# 07
import numpy as np


def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two face embeddings.
    """

    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

    return float(similarity)