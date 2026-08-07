# preprocessing/clean.py

import re

def clean_wikipedia(text: str) -> str:
    """
    Cleans raw Wikipedia API text.
    Removes LaTeX, math symbols, and formatting noise.
    """

    # Remove LaTeX displaystyle blocks
    text = re.sub(r'\{\\displaystyle[^}]*\}', '', text)

    # Remove all remaining LaTeX-style blocks
    text = re.sub(r'\{[^}]{0,100}\}', '', text)

    # Remove lines with math operators and no real words
    text = re.sub(r'[=<>+\-*/\\^_|]{3,}', '', text)

    # Remove unicode math symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove leftover display artifacts like isolated variable names
    # Lines that are just 1-3 chars of math notation
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append('')
            continue

        # Skip very short lines
        if len(stripped) <= 3:
            continue

        # Skip lines with low alphabetic ratio (math lines)
        alpha = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 10 and alpha / len(stripped) < 0.5:
            continue

        # Skip lines that look like math expressions
        if re.search(r'\\[a-zA-Z]+|_{|}\s*\(', stripped):
            continue

        cleaned.append(line)

    text = '\n'.join(cleaned)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from data.fetch import fetch_wikipedia

    raw   = fetch_wikipedia("Transformer (machine learning model)")
    clean = clean_wikipedia(raw)

    print(f"Raw length:     {len(raw):,} chars")
    print(f"Cleaned length: {len(clean):,} chars")
    print(f"Removed:        {len(raw)-len(clean):,} chars")

    print(f"\nFirst 50000 to 55000 chars of Raw Text:")
    print(raw[50000:55000])

    print(f"\nFirst 50000 to 55000 chars of Cleaned Text:")
    print(clean[50000:55000])

    print(f"\nMiddle 300 chars:")
    mid = len(clean) // 2
    print(clean[mid:mid+300])

    # Check for remaining math artifacts
    math_lines = [l for l in clean.split('\n')
                  if re.search(r'\{|\\display|displaystyle', l)]
    print(f"\nRemaining math artifacts: {len(math_lines)} lines")
    if math_lines:
        for l in math_lines[:3]:
            print(f"  {l[:100]}")