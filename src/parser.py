from dataclasses import dataclass
from typing import List
import frontmatter
import re
from src.config import NOTES_PATH
from pathlib import Path
from datetime import date, datetime


@dataclass
class Note:
    title: str
    tags: List[str]
    created: str
    content: str
    links: List[str] # [[Linked Note]] style internal links


def extract_links(text: str) -> List[str]:
    return re.findall(r"\[\[([^\]]+)\]\]", text)

def load_note(file_path: Path) -> Note:
    post = frontmatter.load(file_path)
    content = post.content
    links = extract_links(content)

    created = post.get('created', '')
    if isinstance(created, (date, datetime)):
        created = str(created)

    return Note(
        title=post.get('title', file_path.stem),
        tags=post.get('tags', []),
        created=created,
        content=content,
        links=links
    )


def load_all_notes(path: Path = NOTES_PATH) -> List[Note]:
    return [load_note(f) for f in path.glob('*.md')]