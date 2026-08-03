from app.index.text_processing import tokenize


def test_stopwords_are_removed():
    tokens = tokenize("the cat is on the mat and happy")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "and" not in tokens
    assert "on" not in tokens
    # Content words survive (possibly stemmed).
    assert "cat" in tokens
    assert "mat" in tokens
    assert "happi" in tokens  # Porter stem of "happy"


def test_stemming_collapses_inflections():
    running = tokenize("running")
    runs = tokenize("runs")
    assert running == runs
    assert running == ["run"]


def test_non_alphabetic_tokens_filtered():
    tokens = tokenize(" thruster costs $100 !!! version 2.0 ")
    joined = " ".join(tokens)
    assert "100" not in tokens
    assert "2" not in tokens
    assert "0" not in tokens
    assert "$" not in joined
    assert "!" not in joined
    # Alphabetic content remains (stemmed).
    assert "thruster" in tokens
    assert "cost" in tokens
    assert "version" in tokens


def test_empty_string_returns_empty_list():
    assert tokenize("") == []
    assert tokenize("   ") == []
