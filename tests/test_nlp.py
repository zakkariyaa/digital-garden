from src.nlp import extract_tags, extract_keywords

def test_extract_tags():
    text = "This note discusses Python testing tools like Pytest and Unittest."
    tags = extract_tags(text)
    assert any("pytest" in tag for tag, _ in tags) or any("test" in tag for tag, _ in tags)

def test_extract_keywords():
    text = "Testing tools like Pytest help structure tests."
    keywords = extract_keywords(text)
    assert isinstance(keywords, list)
    assert len(keywords) > 0