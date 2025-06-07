# 🌱 Digital Garden Engine

A command-line tool for managing and exploring a personal digital garden of markdown notes — powered by semantic search, graph theory, and NLP.

---

## ✨ Features

- 🔍 Full-text search and semantic similarity between notes
- 📚 Markdown note parsing with frontmatter (title, tags, created)
- 🧠 Smart tag suggestions and keyword extraction
- 🔗 Bidirectional link detection (`[[Note]]`)
- 🗺 Graph-based note clustering + interactive visualization
- ✍️ CLI commands to create, rename, delete notes
- 📊 Stats dashboard and note analytics
- ✅ Built-in tests + GitHub Actions CI

---

## 🚀 Getting Started

### 🔧 Install

```bash
git clone https://github.com/zakkariyaa/digital-garden.git
cd digital-garden
pip install -e .
```

Then use it like so
```bash
digital-garden list-notes
digital-garden ...
```


## 📁 Note Format
Markdown notes live in notes/ and must start with YAML frontmatter:

```bash
---
title: My Note
tags: [python, testing]
created: 2024-01-01
---

This is my note body. It links to [[Other Note]].
```

## 🛠 CLI Usage
```bash
python main.py list-notes
python main.py create-note "My Note" --tags "dev,ideas"
python main.py suggest-tags "My Note"
python main.py visualize-graph --method embed
python main.py stats
```

For full commands:
```bash
python main.py --help
```

## 🧪 Testing
```
PYTHONPATH=./ pytest
```
