import typer
from .parser import load_all_notes
from .graph import build_graph
from rich import print
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from src.semantic import compute_similarity_tfidf, get_top_related_notes, detect_clusters


app = typer.Typer()

@app.command()
def hello():
    print("🌱 Digital Garden CLI ready!")


@app.command()
def list_notes():
    notes = load_all_notes()
    for note in notes:
        print(f'📒 Note: {note.title}')
        print(f'Tags: {note.tags}')
        print(f'Created: {note.created}')
        print(f'Links: {note.links}')

        print(f'Top Related Notes: ')
        top_related_notes = get_top_related_notes(note, notes, 3)
        for other_title, score in top_related_notes:
            print(f"\t • {other_title} [dim](score: {score:.2f})[/dim]")
        
        print()


@app.command()
def show_graph():
    notes = load_all_notes()
    G = build_graph(notes)

    print("[bold underline green]📚 Notes in Graph:[/]")
    for node in G.nodes:
        print(f"• {node}")

    print("\n[bold underline cyan]🔗 Links (edges):[/]")
    for source, target in G.edges:
        print(f"{source} → {target}")


@app.command()
def connected_to(title: str):
    notes = load_all_notes()
    G = build_graph(notes)

    if title not in G:
        print(f"[red]Note '{title}' not found in graph.[/]")
        return

    links = list(G.successors(title))
    backlinks = list(G.predecessors(title))

    print(f"\n[bold green]{title}[/] links to: ")
    for node in links:
        print(f"→ {node}")

    print(f"\n[bold blue]{title}[/] is linked from:")
    for node in backlinks:
        print(f"← {node}")


@app.command()
def visualize_graph(threshold: float = 0.5, save: bool = False):
    """Render a graph of notes and their links."""
    notes = load_all_notes()
    sim_dict = compute_similarity_tfidf(notes)
    clusters = detect_clusters(sim_dict, threshold)
    G = build_graph(notes)

    # Assign cluster IDs
    note_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for note in cluster:
            note_to_cluster[note] = i

    # Assign colors
    num_clusters = len(clusters)
    color_map = cm.get_cmap("tab10", num_clusters)
    node_colors = [
        color_map(note_to_cluster.get(node, 0)) for node in G.nodes
    ]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=2000,
        font_size=10,
        font_weight="bold"
    )
    plt.title("Digital Garden - Note Clusters")
    plt.tight_layout()

    if save:
        plt.savefig("clustered-note-graph.png")
        print("[blue]📸 Graph saved as clustered-note-graph.png[/]")
    else:
        plt.show()
