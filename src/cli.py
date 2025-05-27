import typer

app = typer.Typer()

@app.command()
def hello():
    print("🌱 Digital Garden CLI ready!")

# Usage: python main.py hello