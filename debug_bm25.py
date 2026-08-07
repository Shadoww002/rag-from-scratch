# save as debug_bm25.py in project root

import sys
import os
sys.path.append(".")

from data.fetch import fetch_wikipedia
from preprocessing.clean import clean_wikipedia
from preprocessing.chunk import chunk_fixed
from retrieval.bm25_retriever import build_bm25_index, tokenize

raw    = fetch_wikipedia("Transformer (machine learning model)")
text   = clean_wikipedia(raw)
chunks = chunk_fixed(text, chunk_size=500, overlap=50)
bm25   = build_bm25_index(chunks)

# Find the winning chunk and the correct chunk
# and show exactly what tokens they share with the query

query        = "What paper introduced the transformer in 2017?"
query_tokens = tokenize(query)
print(f"Query tokens: {query_tokens}\n")

scores      = bm25.get_scores(query_tokens)
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

# Find the correct chunk index
correct_idx = None
for i, chunk in enumerate(chunks):
    if "proposed in the 2017" in chunk:
        correct_idx = i
        break

print(f"Correct chunk index: {correct_idx}")
print(f"Correct chunk score: {scores[correct_idx]:.4f}")
print(f"Correct chunk rank:  {list(top_indices).index(correct_idx) if correct_idx in top_indices else 'not in top 5'}")
print()

# Show token overlap for top 3 chunks vs correct chunk
for rank, idx in enumerate(top_indices[:3]):
    chunk_tokens = set(tokenize(chunks[idx]))
    overlap      = set(query_tokens) & chunk_tokens
    print(f"Rank {rank} (idx={idx}) score={scores[idx]:.4f}")
    print(f"  Overlapping tokens: {overlap}")
    print(f"  Chunk preview: {chunks[idx][:120]}")
    print()

print(f"── Correct chunk (idx={correct_idx}) ──")
chunk_tokens = set(tokenize(chunks[correct_idx]))
overlap      = set(query_tokens) & chunk_tokens
print(f"  Overlapping tokens: {overlap}")
print(f"  Chunk preview: {chunks[correct_idx][:200]}")

# debug_bm25.py — add at bottom

print("\n\n── Testing with smaller chunks (300 chars) ──")
chunks_small = chunk_fixed(text, chunk_size=300, overlap=30)
bm25_small   = build_bm25_index(chunks_small)

scores_small = bm25_small.get_scores(query_tokens)
top_small    = sorted(range(len(scores_small)),
                      key=lambda i: scores_small[i],
                      reverse=True)[:3]

print(f"Total chunks: {len(chunks_small)}")
for rank, idx in enumerate(top_small):
    print(f"\nRank {rank} score={scores_small[idx]:.4f}")
    print(f"  {chunks_small[idx][:200]}")