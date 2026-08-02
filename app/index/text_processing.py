from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def extract_text(html: str) -> tuple[str, str]:
    """
    Pull the page title and visible body text out of raw HTML.

    Returns:
        (title, body_text) — title is "" if the page has no <title> tag.
        <script> and <style> contents are removed before body extraction
        so they never pollute the search index.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.title
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag in soup(["script", "style"]):
        tag.decompose()

    body_text = soup.get_text(separator=" ", strip=True)
    return title, body_text


def tokenize(text: str) -> list[str]:
    """
    Normalize free text into a list of stemmed index terms.

    Pipeline: lowercase → word_tokenize → keep alphabetic tokens only
    → drop English stopwords → Porter-stem what remains.
    """
    tokens = word_tokenize(text.lower())
    return [
        _STEMMER.stem(token)
        for token in tokens
        if token.isalpha() and token not in _STOPWORDS
    ]
