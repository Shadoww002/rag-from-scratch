# generator/answer.py — Generate answer using Groq API
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()   # loads .env file automatically
# Initialize Groq client
# Key is read from environment variable GROQ_API_KEY
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL   = "llama-3.1-8b-instant"# fast, free, strong quality


def generate_answer(
    question: str,
    chunks:   list[dict],
) -> dict:
    """
    Generates an answer using Groq (llama3) with retrieved chunks as context.

    Args:
        question: user's question string
        chunks:   list of dicts with 'text' key from retrieve()

    Returns:
        dict with answer and chunks used
    """
    if not chunks:
        return {
            "answer": "No relevant chunks retrieved.",
            "chunks": [],
        }

    # Build context from retrieved chunks
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"[Chunk {i+1}]\n{c['text']}")

    context = "\n\n".join(context_parts)

    # Prompt — forces model to answer only from context
    prompt = f"""Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I cannot find this in the provided context."
Keep your answer concise and factual — 1-3 sentences maximum.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,      # deterministic — same question same answer
        max_tokens=256,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "chunks": chunks,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.fetch import fetch_wikipedia
    from preprocessing.clean import clean_wikipedia
    from preprocessing.chunk import chunk_fixed
    from embeddings.embed import embed_chunks
    from vectorstore.store import create_collection, store_chunks, retrieve

    # Build pipeline
    print("Building pipeline...")
    raw        = fetch_wikipedia("Transformer (machine learning model)")
    text       = clean_wikipedia(raw)
    chunks     = chunk_fixed(text, chunk_size=500, overlap=50)
    embeddings = embed_chunks(chunks)
    collection = create_collection("rag_generator_test")
    store_chunks(collection, chunks, embeddings)
    print("Pipeline ready.\n")

    questions = [
        "When was the transformer architecture proposed?",
        "What does BERT stand for?",
        "How does the attention mechanism work?",
        "What is the difference between encoder and decoder?",
        "What paper introduced the transformer in 2017?",
    ]

    print("=" * 65)
    print("FULL RAG PIPELINE TEST")
    print("=" * 65)

    for q in questions:
        chunks_retrieved = retrieve(collection, q, n_results=5)
        result           = generate_answer(q, chunks_retrieved)

        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"   top chunk score: {chunks_retrieved[0]['score']} | "
              f"{chunks_retrieved[0]['text'][:70]}...")
        print("-" * 65)