def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Fixed-size character chunking with overlap.

    Args:
        text:       raw text to split
        chunk_size: number of characters per chunk
        overlap:    characters shared between consecutive chunks

    Returns:
        list of text chunks

    Why overlap?
        If an answer sits at a chunk boundary, overlap ensures
        it appears fully in at least one chunk.

        Without overlap:
        Chunk 1: "...The attention mechanism was invented"
        Chunk 2: "by Vaswani et al. in 2017..."
        Query "who invented attention" → neither chunk has full answer

        With overlap:
        Chunk 1: "...The attention mechanism was invented"
        Chunk 2: "attention mechanism was invented by Vaswani et al. in 2017..."
        Query "who invented attention" → chunk 2 has complete answer
    """
    chunks = []
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # step forward by (chunk_size - overlap)
        # so next chunk starts 'overlap' chars before current chunk ended
        start = end - overlap

    return chunks


def chunk_stats(chunks: list[str]) -> None:
    """Print basic statistics about the chunks."""
    lengths = [len(c) for c in chunks]
    
    print(f"Total chunks:     {len(chunks)}")
    print(f"Avg chunk length: {sum(lengths) // len(lengths)} chars")
    print(f"Min chunk length: {min(lengths)} chars")
    print(f"Max chunk length: {max(lengths)} chars")


if __name__ == "__main__":
    # Test it directly — run: python -m preprocessing.chunk
    import sys
    sys.path.append(".")
    
    from data.fetch import fetch_wikipedia
    
    text   = fetch_wikipedia("Transformer (machine learning model)")
    chunks = chunk_fixed(text, chunk_size=1000, overlap=200)
    
    chunk_stats(chunks)
    
    print(f"\n── Chunk 0 ──────────────────────────────────")
    print(chunks[0])
    
    print(f"\n── Chunk 1 ──────────────────────────────────")
    print(chunks[1])
    
    print(f"\n── Last 50 chars of chunk 0 ─────────────────")
    print(repr(chunks[0][-50:]))
    
    print(f"\n── First 50 chars of chunk 1 ────────────────")
    print(repr(chunks[1][:50]))
    
    print("\nNotice: chunk 1 starts with the last 50 chars of chunk 0 — that's the overlap working.")