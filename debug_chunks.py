# save as debug_chunks.py in project root
import sys
sys.path.append(".")

from data.fetch import fetch_wikipedia
from preprocessing.clean import clean_wikipedia
from preprocessing.chunk import chunk_fixed

raw    = fetch_wikipedia("Transformer (machine learning model)")
text   = clean_wikipedia(raw)
chunks = chunk_fixed(text, chunk_size=500, overlap=50)

# Find which chunk contains the answer
print("Searching for 'proposed' across all chunks:\n")
for i, chunk in enumerate(chunks):
    if "proposed" in chunk.lower() and "2017" in chunk:
        print(f"── Chunk {i} ──────────────────────────")
        print(chunk)
        print()