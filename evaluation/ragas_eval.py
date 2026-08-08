# evaluation/ragas_eval.py — RAGAS evaluation harness

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL   = "llama-3.1-8b-instant"


# ── Individual Metrics ────────────────────────────────────────
def score_faithfulness(
    answer:  str,
    context: str,
) -> float:
    """
    Measures if every claim in the answer is supported by context.
    Returns 0.0 to 1.0.
    """
    prompt = f"""You are evaluating if an answer is faithful to a given context.

Context:
{context[:600]}

Answer:
{answer}

Task: Check if the answer contains only information that is present in the context.

Return a single decimal number between 0.0 and 1.0:
- 1.0 means every claim in the answer is supported by the context
- 0.5 means about half the claims are supported
- 0.0 means no claims are supported by the context

Return ONLY the number. Example outputs: 1.0 or 0.5 or 0.0"""

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )

    raw = response.choices[0].message.content.strip()

    # Debug — print what LLM actually returned
    print(f"    [faithfulness raw]: '{raw}'")

    # Try to extract a float from the response
    import re
    matches = re.findall(r'\d+\.?\d*', raw)
    if matches:
        val = float(matches[0])
        # Handle case where LLM returns "10" meaning 1.0
        if val > 1.0:
            val = val / 10.0
        return round(min(val, 1.0), 4)

    return 0.0
def score_answer_relevance(
    question: str,
    answer:   str,
) -> float:
    """
    Measures if the answer directly addresses the question.
    Returns 0.0 to 1.0.
    """
    prompt = f"""You are evaluating if an answer is relevant to a question.

Question: {question}
Answer:   {answer}

Instructions:
- 1.0: Answer directly and completely addresses the question
- 0.5: Answer partially addresses the question
- 0.0: Answer does not address the question at all

Return ONLY a number between 0.0 and 1.0. Nothing else."""

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )

    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0


def score_context_precision(
    question: str,
    chunks:   list[str],
) -> float:
    """
    Measures what fraction of retrieved chunks are relevant to the question.
    Returns 0.0 to 1.0.
    """
    if not chunks:
        return 0.0

    relevant = 0
    for chunk in chunks:
        prompt = f"""Is this chunk relevant to answering the question?

Question: {question}
Chunk: {chunk[:300]}

Return ONLY 1 (relevant) or 0 (not relevant). Nothing else."""

        response = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )

        try:
            relevant += int(response.choices[0].message.content.strip())
        except:
            pass

    return round(relevant / len(chunks), 4)


def score_context_recall(
    question:       str,
    chunks:         list[str],
    ground_truth:   str,
) -> float:
    """
    Measures if retrieved chunks contain enough to produce the ground truth answer.
    Returns 0.0 to 1.0.
    """
    context = "\n".join(chunks)

    prompt = f"""You are checking if a context contains enough information
to answer a question correctly.

Question:     {question}
Ground Truth: {ground_truth}
Context:      {context[:800]}

Instructions:
- 1.0: Context contains all information needed for ground truth answer
- 0.5: Context contains partial information
- 0.0: Context does not contain the information needed

Return ONLY a number between 0.0 and 1.0. Nothing else."""

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )

    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0


# ── Full Evaluation Harness ───────────────────────────────────


def evaluate_pipeline(
    pipeline_fn:  callable,
    test_cases:   list[dict],
    verbose:      bool = True,
) -> dict:
    """
    Runs RAGAS evaluation on a full pipeline.

    Args:
        pipeline_fn:  function that takes (question) and returns
                      {"answer": str, "chunks": list[dict]}
        test_cases:   list of dicts:
                        question:     str
                        ground_truth: str  ← correct answer for recall scoring
        verbose:      print per-question results

    Returns:
        dict with average scores across all test cases
    """
    results = []

    for tc in test_cases:
        question     = tc["question"]
        ground_truth = tc["ground_truth"]

        # Run pipeline
        output  = pipeline_fn(question)
        answer  = output["answer"]
        chunks  = [c["text"] for c in output["chunks"]]
        context = "\n".join(chunks)

        # Score all 4 metrics
        faithfulness      = score_faithfulness(answer, context)
        answer_relevance  = score_answer_relevance(question, answer)
        context_precision = score_context_precision(question, chunks)
        context_recall    = score_context_recall(question, chunks, ground_truth)

        result = {
            "question":          question,
            "answer":            answer,
            "faithfulness":      faithfulness,
            "answer_relevance":  answer_relevance,
            "context_precision": context_precision,
            "context_recall":    context_recall,
        }
        results.append(result)

        if verbose:
            print(f"\nQ: {question}")
            print(f"A: {answer[:100]}...")
            print(f"   Faithfulness:      {faithfulness:.2f}")
            print(f"   Answer Relevance:  {answer_relevance:.2f}")
            print(f"   Context Precision: {context_precision:.2f}")
            print(f"   Context Recall:    {context_recall:.2f}")

    # Average scores
    avg = {
        "faithfulness":      round(sum(r["faithfulness"]      for r in results) / len(results), 4),
        "answer_relevance":  round(sum(r["answer_relevance"]  for r in results) / len(results), 4),
        "context_precision": round(sum(r["context_precision"] for r in results) / len(results), 4),
        "context_recall":    round(sum(r["context_recall"]    for r in results) / len(results), 4),
    }

    return {"per_question": results, "averages": avg}


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from data.fetch import fetch_wikipedia
    from preprocessing.clean import clean_wikipedia
    from preprocessing.chunk import chunk_fixed
    from embeddings.embed import embed_chunks
    from vectorstore.store import create_collection, store_chunks
    from retrieval.bm25_retriever import build_bm25_index
    from retrieval.hybrid import retrieve_hybrid
    from retrieval.reranker import rerank
    from generator.answer import generate_answer

    # Build pipeline
    print("Building pipeline...")
    raw        = fetch_wikipedia("Transformer (machine learning model)")
    text       = clean_wikipedia(raw)
    chunks     = chunk_fixed(text, chunk_size=300, overlap=30)
    embeddings = embed_chunks(chunks)
    collection = create_collection("rag_eval")
    store_chunks(collection, chunks, embeddings)
    bm25       = build_bm25_index(chunks)
    print(f"Ready. {len(chunks)} chunks.\n")

    # Define pipeline function
    def full_pipeline(question: str) -> dict:
        candidates = retrieve_hybrid(
            bm25, chunks, collection, question, n_results=20
        )
        reranked = rerank(question, candidates, n_results=5)
        result   = generate_answer(question, reranked)
        return result

    # Test cases with ground truth
    test_cases = [
        {
            "question":     "When was the transformer architecture proposed?",
            "ground_truth": "The transformer architecture was proposed in 2017 in the paper Attention Is All You Need.",
        },
        {
            "question":     "What does BERT stand for?",
            "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers.",
        },
        {
            "question":     "How does the attention mechanism work?",
            "ground_truth": "The attention mechanism computes attention scores as dot products between query and key vectors, scales by square root of dimension, applies softmax, then takes weighted sum of value vectors.",
        },
        {
            "question":     "What is the difference between encoder and decoder?",
            "ground_truth": "The encoder processes the entire input sequence at once while the decoder generates output tokens sequentially attending to encoder output.",
        },
        {
            "question":     "What paper introduced the transformer in 2017?",
            "ground_truth": "The paper Attention Is All You Need introduced the transformer architecture in 2017.",
        },
    ]

    print("=" * 65)
    print("RAGAS EVALUATION — HYBRID + RERANKER PIPELINE")
    print("=" * 65)

    eval_results = evaluate_pipeline(full_pipeline, test_cases, verbose=True)

    print("\n" + "=" * 65)
    print("AVERAGE SCORES")
    print("=" * 65)
    for metric, score in eval_results["averages"].items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.4f}  {bar}")

    print("\nThese are your baseline numbers.")
    print("Every future change gets measured against these.")