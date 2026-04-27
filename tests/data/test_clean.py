import pandas as pd

from src.data.clean import clean_dataframe, clean_text


def test_clean_text_strips_urls_and_emails():
    t = "hello http://x.com and me@me.com"
    assert clean_text(t) == "hello <URL> and <EMAIL>"


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\n\tc") == "a b c"


def test_clean_dataframe_drops_rows_below_min_length():
    df = pd.DataFrame({"text": ["ok", "x"], "source": ["a", "b"]})
    out = clean_dataframe(df, min_len=2)
    assert len(out) == 1
