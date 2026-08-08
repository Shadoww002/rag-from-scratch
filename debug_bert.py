# save as debug_bert.py
import sys
sys.path.append(".")

from data.fetch import fetch_wikipedia
from preprocessing.clean import clean_wikipedia
from preprocessing.chunk import chunk_fixed

raw    = fetch_wikipedia("Transformer (machine learning model)")
text   = clean_wikipedia(raw)
chunks = chunk_fixed(text, chunk_size=300, overlap=30)

print(f"Total chunks: {len(chunks)}\n")
print("Searching for BERT definition chunks:\n")

for i, chunk in enumerate(chunks):
    if "bidirectional" in chunk.lower() or \
       ("bert" in chunk.lower() and "encoder" in chunk.lower()):
        print(f"── Chunk {i} ──────────────────────")
        print(chunk)
        print()