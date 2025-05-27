import typer
from .parser import load_all_notes

app = typer.Typer()

@app.command()
def hello():
    print("🌱 Digital Garden CLI ready!")


@app.command()
def list_notes():
    notes = load_all_notes()
    for note in notes:
        print(f'{note.title} - Tags: {note.tags} - Links: {note.links}')