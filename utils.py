"""
utils.py
Text cleaning helpers for the Fake News Detector project.
Kept dependency-free (no nltk download needed) so it runs anywhere.
"""

import re

# A small, common English stopword list (avoids needing an nltk download)
STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on
for with as by at from this that these those it its it's he she they
them his her their our your my i you we he's she's im ive youre theyre
not no nor so than too very can will just don dont should now
""".split())


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/numbers, and remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"<.*?>", " ", text)                  # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                # punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)
