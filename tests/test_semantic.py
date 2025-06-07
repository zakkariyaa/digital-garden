from pathlib import Path
from src.parser import Note
from src.semantic import (
    compute_similarity_tfidf,
    get_top_related_notes,
    build_similarity_graph,
    detect_clusters,
)

# Sample notes for testing
note1 = Note(
    title="Note A",
    tags=["testing"],
    created="2024-01-01",
    content="This note is about Python testing and pytest.",
    links=[]
)

note2 = Note(
    title="Note B",
    tags=["testing"],
    created="2024-01-02",
    content="Unit testing in Python with unittest and mock.",
    links=[]
)

note3 = Note(
    title="Note C",
    tags=["plants"],
    created="2024-01-03",
    content="This note is about indoor plants and sunlight.",
    links=[]
)

notes = [note1, note2, note3]


def test_compute_similarity_tfidf_creates_cache_and_dict():
    path = Path("data/similarities.json")
    if path.exists():
        path.unlink()

    similarities = compute_similarity_tfidf(notes)
    assert isinstance(similarities, dict)
    assert "Note A" in similarities
    assert path.exists()


def test_get_top_related_notes_returns_top_matches():
    top = get_top_related_notes(note1, notes, n=2)
    assert isinstance(top, list)
    assert len(top) == 2
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in top)


def test_build_similarity_graph_and_detect_clusters():
    similarities = compute_similarity_tfidf(notes)
    G = build_similarity_graph(similarities, threshold=0.1)
    assert G.has_node("Note A")
    assert G.has_node("Note C")

    clusters = detect_clusters(similarities, threshold=0.1)
    assert isinstance(clusters, list)
    assert any("Note A" in group for group in clusters)
    assert any("Note C" in group for group in clusters)


def teardown_module(module):
    """Cleanup similarity cache after tests."""
    path = Path("data/similarities.json")
    if path.exists():
        path.unlink()

