from src.parser import load_note, extract_links, Note
from pathlib import Path


def test_load_note():
    path = Path(__file__).parent / 'fixtures' / 'test_note.md'
    note = load_note(path)

    assert isinstance(note, Note)
    assert note.title == 'Test Note'
    assert note.tags == ['test', 'parser']
    assert note.created == '2023-01-01'
    assert 'This is a test note.' in note.content
    assert set(note.links) == {'Another Note', 'Second Note'}

def test_extract_links():
    text = 'Some text with [[Link One]] and [[Another Link]].'
    links = extract_links(text)

    assert links == ['Link One', 'Another Link']