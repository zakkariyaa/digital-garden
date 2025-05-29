import typer
from .parser import load_all_notes
from .graph import build_graph
from rich import print
import matplotlib.pyplot as plt
import networkx as nx
from src.semantic import compute_similarity_tfidf, compute_similarity_embeddings


app = typer.Typer()

@app.command()
def hello():
    print("🌱 Digital Garden CLI ready!")


@app.command()
def list_notes():
    notes = load_all_notes()
    for note in notes:
        print(f'{note.title} - Tags: {note.tags} - Links: {note.links}')


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

    print(f"\n[bold green]{title}[/] links to:")
    for node in links:
        print(f"→ {node}")

    print(f"\n[bold blue]{title}[/] is linked from:")
    for node in backlinks:
        print(f"← {node}")


@app.command()
def visualize_graph(save: bool = False):
    """Render a graph of notes and their links."""
    notes = load_all_notes()
    graph = build_graph(notes)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightgreen",
        edge_color="gray",
        node_size=2000,
        font_size=10,
        font_weight="bold",
    )

    plt.title("Digital Garden Note Graph")
    plt.tight_layout()
    print("[green]📈 Showing graph window...[/]")

    if save:
        plt.savefig("note-graph.png")
        print("[blue]📸 Graph saved as note-graph.png[/]")
    else:
        plt.show()


@app.command()
def suggest_related(title: str, method: str = "tfidf"):
    """
    Suggest notes similar to the given note title.
    --method: 'tfidf' (default) or 'embed'
    """
    notes = load_all_notes()
    sim_func = compute_similarity_tfidf if method == "tfidf" else compute_similarity_embeddings

    similarities = sim_func(notes)

    if title not in similarities:
        print(f"[red]Note titled '{title}' not found.[/red]")
        return

    print(f"[bold green]Related notes for:[/bold green] {title}")
    for other_title, score in similarities[title][:5]:
        print(f" • {other_title} [dim](score: {score:.2f})[/dim]")
