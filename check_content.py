# save as check_content.py in project root

import sys
sys.path.append(".")

from data.fetch import fetch_wikipedia
from preprocessing.clean import clean_wikipedia

raw   = fetch_wikipedia("Transformer (machine learning model)")
text  = clean_wikipedia(raw)

# Search for key terms to see what language the article uses
search_terms = [
    "proposed",
    "invented",
    "Vaswani",
    "BERT",
    "stands for",
    "attention is all",
    "2017",
]

print("Searching cleaned text for key terms:\n")
for term in search_terms:
    idx = text.lower().find(term.lower())
    if idx == -1:
        print(f"'{term}': NOT FOUND in article")
    else:
        # show surrounding context
        start = max(0, idx - 100)
        end   = min(len(text), idx + 150)
        print(f"'{term}' found at position {idx}:")
        print(f"  ...{text[start:end]}...")
        print()