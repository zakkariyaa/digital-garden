from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

from src.parser import Note


def compute_similarity_tfidf(notes: List[Note]) -> Dict[str, List[Tuple[str, float]]]:
    titles = [note.title for note in notes]
    contents = [note.content for note in notes]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    sim_matrix = cosine_similarity(tfidf_matrix)

    return _build_similarity_dict(titles, sim_matrix)


def compute_similarity_embeddings(notes: List[Note]) -> Dict[str, List[Tuple[str, float]]]:
    titles = [note.title for note in notes]
    contents = [note.content for note in notes]

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small + fast
    embeddings = model.encode(contents, convert_to_tensor=True)
    sim_matrix = cosine_similarity(embeddings.cpu().numpy())

    return _build_similarity_dict(titles, sim_matrix)


def _build_similarity_dict(titles: List[str], sim_matrix: np.ndarray) -> Dict[str, List[Tuple[str, float]]]:
    similarity_dict = {}
    for i, title in enumerate(titles):
        scores = [
            (titles[j], float(sim_matrix[i][j]))
            for j in range(len(titles)) if i != j
        ]
        similarity_dict[title] = sorted(scores, key=lambda x: x[1], reverse=True)
    return similarity_dict
