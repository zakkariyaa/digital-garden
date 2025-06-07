from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import spacy
import yake
from typing import Tuple


def summarise(text: str) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)
    return " ".join(str(s) for s in summary)


_nlp_model = None

def extract_tags(text: str, top_n: int = 5) -> list[str]:
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm")

    doc = _nlp_model(text.lower())
    
    # Extract noun chunks (phrases) and named entities
    tags = set()

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            if 3 <= len(token.text) <= 25:
                tags.add(token.lemma_.lower())

    for ent in doc.ents:
        tags.add(ent.text.strip())

    # Optionally: frequency sort
    freq = {}
    for token in doc:
        if token.text in tags:
            freq[token.text] = freq.get(token.text, 0) + 1

    return sorted(freq.items(), key=lambda x: -x[1])[:top_n]


def extract_keywords(text: str, top_n: int = 5) -> list[Tuple[str, float]]:
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=top_n)
    keywords = kw_extractor.extract_keywords(text)
    return keywords  # List of (keyword, score)
