# evaluation/compare.py — Compare all pipeline variants

import sys
sys.path.append(".")

from data.fetch import fetch_wikipedia
from preprocessing.clean import clean_wikipedia
from preprocessing.chunk import chunk_fixed
from embeddings.embed import embed_chunks
from vectorstore.store import create_collection, store_chunks, retrieve as dense_retrieve
from retrieval.bm25_retriever import build_bm25_index, retrieve_bm25
from retrieval.hybrid import retrieve_hybrid
from retrieval.reranker import rerank
from generator.answer import generate_answer
from evaluation.ragas_eval import evaluate_pipeline

# ── Build corpus once ─────────────────────────────────
print("Building corpus...")
raw        = fetch_wikipedia("Transformer (machine learning model)")
text       = clean_wikipedia(raw)
chunks     = chunk_fixed(text, chunk_size=300, overlap=30)
embeddings = embed_chunks(chunks)
collection = create_collection("rag_compare")
store_chunks(collection, chunks, embeddings)
bm25       = build_bm25_index(chunks)
print(f"Ready. {len(chunks)} chunks.\n")

# ── Test cases ────────────────────────────────────────
test_cases = [
    {
        "question":     "When was the transformer architecture proposed?",
        "ground_truth": "The transformer was proposed in 2017 in Attention Is All You Need.",
    },
    {
        "question":     "What does BERT stand for?",
        "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers.",
    },
    {
        "question":     "How does the attention mechanism work?",
        "ground_truth": "Attention computes dot products between query and key vectors scaled by sqrt of dimension then applies softmax to get weights for value vectors.",
    },
    {
        "question":     "What is the difference between encoder and decoder?",
        "ground_truth": "Encoder processes entire input at once while decoder generates output sequentially.",
    },
    {
        "question":     "What paper introduced the transformer in 2017?",
        "ground_truth": "Attention Is All You Need introduced the transformer in 2017.",
    },
]

# ── Pipeline variants ─────────────────────────────────

def pipeline_dense(q):
    chunks_r = dense_retrieve(collection, q, n_results=5)
    return generate_answer(q, chunks_r)

def pipeline_bm25(q):
    chunks_r = retrieve_bm25(bm25, chunks, q, n_results=5)
    return generate_answer(q, chunks_r)

def pipeline_hybrid(q):
    chunks_r = retrieve_hybrid(bm25, chunks, collection, q, n_results=5)
    return generate_answer(q, chunks_r)

def pipeline_hybrid_rerank(q):
    candidates = retrieve_hybrid(bm25, chunks, collection, q, n_results=20)
    reranked   = rerank(q, candidates, n_results=5)
    return generate_answer(q, reranked)

pipelines = {
    "1. Dense only":       pipeline_dense,
    "2. BM25 only":        pipeline_bm25,
    "3. Hybrid":           pipeline_hybrid,
    "4. Hybrid + Rerank":  pipeline_hybrid_rerank,
}

# ── Run all pipelines ─────────────────────────────────
all_results = {}

for name, fn in pipelines.items():
    print(f"\nEvaluating: {name}")
    print("-" * 40)
    results = evaluate_pipeline(fn, test_cases, verbose=False)
    all_results[name] = results["averages"]
    avg = results["averages"]
    print(f"  faithfulness:      {avg['faithfulness']:.4f}")
    print(f"  answer_relevance:  {avg['answer_relevance']:.4f}")
    print(f"  context_precision: {avg['context_precision']:.4f}")
    print(f"  context_recall:    {avg['context_recall']:.4f}")

# ── Final comparison table ────────────────────────────
print("\n\n" + "=" * 75)
print("FINAL COMPARISON — ALL PIPELINES")
print("=" * 75)
print(f"{'Pipeline':<25} {'Faith':>8} {'AnsRel':>8} {'CtxPrec':>8} {'CtxRec':>8}")
print("-" * 75)

for name, scores in all_results.items():
    print(
        f"{name:<25} "
        f"{scores['faithfulness']:>8.4f} "
        f"{scores['answer_relevance']:>8.4f} "
        f"{scores['context_precision']:>8.4f} "
        f"{scores['context_recall']:>8.4f}"
    )

print("=" * 75)
print("\nHigher is better for all metrics.")
print("Compare rows to see what each component added.")