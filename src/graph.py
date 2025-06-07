from networkx import DiGraph
from typing import List
from src.parser import Note


def build_graph(notes: List[Note]) -> DiGraph:
    G = DiGraph()
    for note in notes:
        G.add_node(note.title)
        for link in note.links:
            G.add_edge(note.title, link)
    return G
