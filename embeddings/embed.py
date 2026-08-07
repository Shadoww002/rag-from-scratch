# embeddings/embed.py — Convert text chunks into embedding vectors

from sentence_transformers import SentenceTransformer

# Load once at module level — expensive to reload every call
# all-MiniLM-L6-v2:
#   - 90MB model
#   - 384-dimensional embeddings
#   - Fast on CPU
#   - Good quality for retrieval tasks
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Converts a list of text chunks into embedding vectors.

    Args:
        chunks: list of text strings

    Returns:
        list of embedding vectors — each vector is 384 floats

    normalize_embeddings=True:
        Scales every vector to length 1.
        Required for cosine similarity to work correctly.
        Without it, longer texts get artificially higher scores
        just because their vectors are bigger — not because
        they're more relevant.
    """
    embeddings = _model.encode(
        chunks,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Converts a single query string into an embedding vector.

    Separate function from embed_chunks because:
    - query is a single string, not a list
    - no progress bar needed
    - called at query time, not indexing time
    """
    embedding = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding[0].tolist()


def embedding_stats(embeddings: list[list[float]]) -> None:
    """Print basic statistics about the embeddings."""
    dim = len(embeddings[0])
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dim:    {dim}")
    print(f"Sample vector:    [{embeddings[0][0]:.4f}, "
          f"{embeddings[0][1]:.4f}, "
          f"{embeddings[0][2]:.4f}, ...]")


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from data.fetch import fetch_wikipedia
    from preprocessing.chunk import chunk_fixed

    print("Fetching text...")
    text   = fetch_wikipedia("Transformer (machine learning model)")
    chunks = chunk_fixed(text, chunk_size=500, overlap=50)

    print(f"\nEmbedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    print()
    embedding_stats(embeddings)

    # Verify normalization — length of each vector should be ~1.0
    import math
    vec    = embeddings[0]
    length = math.sqrt(sum(x**2 for x in vec))
    print(f"\nVector length (should be ~1.0): {length:.6f}")

    # Show similarity between two related chunks
    def cosine_similarity(a, b):
        return sum(x*y for x, y in zip(a, b))

    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_0_last = cosine_similarity(embeddings[0], embeddings[-1])

    print(f"\nSimilarity chunk 0 vs chunk 1 (adjacent):  {sim_01:.4f}")
    print(f"Similarity chunk 0 vs last chunk (distant): {sim_0_last:.4f}")
    print("Adjacent chunks should be more similar than distant ones.")