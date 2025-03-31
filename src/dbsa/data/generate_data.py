import re
from collections import OrderedDict

# AISTATS experiments


def clean_sentence(sentence):
    # Remove punctuation and numbers
    cleaned = re.sub(r"[^\w\s]|\d", "", sentence)
    # Split into words and filter out empty strings
    return [word for word in cleaned.split() if word]


def eval_prompt(sentence):
    return "Evaluate this sentence: " + sentence


# If you want to generate the neighbors automatically
# But the word embedding model is usually not very good
# def get_n_nearest_neighbors(word, n=20):
#     try:
#         top_neighbors = model.wv.most_similar(word, topn=n)
#         neighbors = [n for n, s in top_neighbors]
#         return neighbors
#     except KeyError:
#         print(f"The word '{word}' is not in the vocabulary.")
#         return []


def tokenize_and_prepare_for_scoring(text):
    tokens = re.findall(r"\$?\d+(?:,\d+)*(?:\.\d+)?|\w+|[^\w\s]", text)
    token_positions = OrderedDict()
    for index, token in enumerate(tokens):
        if token not in token_positions:
            token_positions[token] = {"positions": [], "score": None, "pval": None}
        token_positions[token]["positions"].append(index)
    return tokens, token_positions
