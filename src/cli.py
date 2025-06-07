import os
import frontmatter
import typer
from typing import Optional
from pathlib import Path
from rich import print
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from datetime import datetime
from collections import Counter
from src.parser import load_all_notes
from src.graph import build_graph
from src.semantic import compute_similarity_tfidf, get_top_related_notes, detect_clusters
from src.nlp import summarise, extract_tags, extract_keywords
from src.graph import build_graph


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


@app.command()
def summarise_note(title: str, method: str = "transformer"):
    """Summarise a single note by title using sumy."""
    notes = load_all_notes()
    note = next((n for n in notes if n.title.lower() == title.lower()), None)
    if not note:
        print(f"[red]Note titled '{title}' not found.[/red]")
        return

    print(f"[bold green]Summary for:[/] {note.title} ({method})\n")
    summary = summarise(note.content)
    print(summary)


@app.command()
def suggest_tags(title: str):
    """Suggest tags for a given note based on its content."""
    notes = load_all_notes()
    note = next((n for n in notes if n.title.lower() == title.lower()), None)
    if not note:
        print(f"[red]Note titled '{title}' not found.[/red]")
        return

    print(f"[bold green]Suggested tags for:[/] {note.title}")
    tags = extract_tags(note.content)
    for tag, freq in tags:
        print(f"• {tag} [dim](freq: {freq})[/dim]")


@app.command()
def extract_keywords_cmd(title: str):
    """Extract top keywords from note content using YAKE."""
    notes = load_all_notes()
    note = next((n for n in notes if n.title.lower() == title.lower()), None)
    if not note:
        print(f"[red]Note titled '{title}' not found.[/red]")
        return

    print(f"[bold green]Keywords for:[/] {note.title}")
    keywords = extract_keywords(note.content)
    for word, score in keywords:
        print(f"• {word} [dim](score: {score:.4f})[/dim]")


@app.command()
def create_note(title: str, tags: Optional[str] = "", folder: str = "notes"):
    """Create a new markdown note with frontmatter."""
    safe_title = title.strip().replace(" ", "_")
    filename = f"{safe_title}.md"
    path = Path(folder) / filename

    if path.exists():
        print(f"[red]Note '{filename}' already exists.[/red]")
        return

    created = datetime.now().date().isoformat()
    tags_list = [t.strip() for t in tags.split(",") if t.strip()]

    frontmatter = [
        "---",
        f"title: {title}",
        f"tags: {tags_list}",
        f"created: {created}",
        "---\n\n"
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(frontmatter))
        f.write("Write your note here...")

    print(f"[green]✅ Created:[/] {path}")


@app.command()
def rename_note(old_title: str, new_title: str):
    """Rename a note by title."""
    notes = load_all_notes()
    old_note = next((n for n in notes if n.title.lower() == old_title.lower()), None)
    if not old_note:
        print(f"[red]Note titled '{old_title}' not found.[/red]")
        return

    notes_dir = Path("notes")
    for file in notes_dir.glob("*.md"):
        loaded = frontmatter.load(file)
        if loaded.get("title", "").lower() == old_title.lower():
            new_filename = new_title.strip().replace(" ", "_") + ".md"
            new_path = notes_dir / new_filename
            file.rename(new_path)

            # Optionally update the title in frontmatter
            loaded["title"] = new_title
            with open(new_path, "w") as f:
                f.write(frontmatter.dumps(loaded))

            print(f"[green]✅ Renamed:[/] {file.name} → {new_path.name}")
            return

    print("[red]Original file not found in note directory.[/red]")


@app.command()
def delete_note(title: str, confirm: bool = False):
    """Delete a note by title (requires --confirm to proceed)."""
    notes = load_all_notes()
    note = next((n for n in notes if n.title.lower() == title.lower()), None)
    if not note:
        print(f"[red]Note titled '{title}' not found.[/red]")
        return

    file_path = next((Path("notes") / f for f in os.listdir("notes") if f.lower().endswith(".md") and title.lower() in f.lower()), None)
    if not file_path:
        print("[red]File not found.[/red]")
        return

    if not confirm:
        print(f"[yellow]Add --confirm to actually delete '{file_path.name}'[/yellow]")
        return

    file_path.unlink()
    print(f"[red]❌ Deleted:[/] {file_path.name}")


@app.command()
def stats():
    """Show garden-level stats and analytics."""
    notes = load_all_notes()
    total = len(notes)
    total_words = sum(len(n.content.split()) for n in notes)
    avg_words = total_words / total if total else 0

    all_tags = [tag for note in notes for tag in note.tags]
    tag_counts = Counter(all_tags).most_common()

    G = build_graph(notes)
    degrees = G.degree()
    top_connected = sorted(degrees, key=lambda x: -x[1])[:5]
    unlinked = [n.title for n in notes if G.degree(n.title) == 0]

    print(f"[bold green]📊 Digital Garden Stats[/bold green]")
    print(f"• Total notes: {total}")
    print(f"• Avg word count: {avg_words:.1f}")
    print(f"• Total tags: {len(set(all_tags))}")
    print(f"• Top tags:")
    for tag, count in tag_counts[:5]:
        print(f"   - {tag} ({count})")

    print("\n• Most linked notes:")
    for title, degree in top_connected:
        print(f"   - {title} ({degree} links)")

    print("\n• Orphan notes (no links):")
    for title in unlinked:
        print(f"   - {title}")
