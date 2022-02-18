from string import punctuation
PUNCTUATION = set(punctuation)

def tokenize_lemmatized(text):
    return [
        token
        for token in text.split(" ")
        if token not in PUNCTUATION or not token.isdigit()
    ]

def count_token(token, text, rel=False):
    """
    Assumes lemmatized text.
    """
    tokens = list(map(lambda x: x.lower(), tokenize_lemmatized(text)))
    return tokens.count(token) if not rel else tokens.count(token) / len(tokens)

def sentenize_dataset(df):
    pass

def get_stopwords(path="../data/stopwords-de.txt"):
    with open(path, encoding="utf-8") as f:
        stopwords = f.read().split()
    return set(stopwords)


