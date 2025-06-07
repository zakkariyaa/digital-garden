import json
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import networkx as nx
from networkx.algorithms.community import label_propagation_communities
from src.parser import Note


def compute_similarity_tfidf(notes: List[Note]) -> Dict[str, List[Tuple[str, float]]]:
    cache_path = Path("data/similarities.json")

    if cache_path.exists() and cache_path.stat().st_size > 0:
        with open(cache_path) as f:
            return {
                k: [(title, float(score)) for title, score in v]
                for k, v in json.load(f).items()
            }

    titles = [note.title for note in notes]
    contents = [note.content for note in notes]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    sim_matrix = cosine_similarity(tfidf_matrix)

    result = _build_similarity_dict(titles, sim_matrix)

    # Save to cache
    safe_data = {
        k: [[title, score] for title, score in v]
        for k, v in result.items()
    }
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(safe_data, f, indent=2)

    return result


# def compute_similarity_embeddings(notes: List[Note]) -> Dict[str, List[Tuple[str, float]]]:
#     titles = [note.title for note in notes]
#     contents = [note.content for note in notes]

#     model = SentenceTransformer('all-MiniLM-L6-v2')  # Small + fast
#     embeddings = model.encode(contents, convert_to_tensor=True)
#     sim_matrix = cosine_similarity(embeddings.cpu().numpy())

#     return _build_similarity_dict(titles, sim_matrix)


def _build_similarity_dict(titles: List[str], sim_matrix: np.ndarray) -> Dict[str, List[Tuple[str, float]]]:
    similarity_dict = {}
    for i, title in enumerate(titles):
        scores = [
            (titles[j], float(sim_matrix[i][j]))
            for j in range(len(titles)) if i != j
        ]
        similarity_dict[title] = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return similarity_dict


def _save_similarity_cache(similarity_dict: Dict[str, List[Tuple[str, float]]]):
    path = Path("data") / "similarities.json"

    # Convert tuples to lists for JSON compatibility
    safe_data = {
        k: [[title, score] for title, score in v]
        for k, v in similarity_dict.items()
    }

    with open(path, "w") as f:
        json.dump(safe_data, f, indent=2)


def _load_similarity_cache() -> Optional[Dict[str, List[Tuple[str, float]]]]:
    path = Path("data") / "similarities.json"
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path) as f:
        return {k: [(title, float(score)) for title, score in v] for k, v in json.load(f).items()}


def get_top_related_notes(note: Note, all_notes: List[Note], n: int = 3) -> List[Tuple[str, float]]:
    similarities = _load_similarity_cache()
    if not similarities:
        similarities = compute_similarity_tfidf(all_notes)
        _save_similarity_cache(similarities)
    else:
        similarities = compute_similarity_tfidf(all_notes)
        _save_similarity_cache(similarities)

    if note.title not in similarities:
        print(f"[red]Note titled '{note.title}' not found.[/red]")
        return

    return similarities[note.title][:n]


def build_similarity_graph(similarity_dict: Dict[str, List[Tuple[str, float]]], threshold: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    for source, neighbors in similarity_dict.items():
        G.add_node(source)
        for target, score in neighbors:
            if score >= threshold:
                G.add_edge(source, target, weight=score)
    return G


def detect_clusters(similarity_dict: Dict[str, List[Tuple[str, float]]], threshold: float = 0.5) -> List[List[str]]:
    G = build_similarity_graph(similarity_dict, threshold)
    communities = label_propagation_communities(G)
    return [sorted(list(group)) for group in communities]

