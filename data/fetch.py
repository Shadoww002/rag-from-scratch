import requests

def fetch_wikipedia(title: str) -> str:
    url = "https://en.wikipedia.org/w/api.php"

    params = {
    "action": "query",
    "format": "json",
    "titles": title,
    "prop": "extracts",
    "explaintext": True,
    # "exintro": True,
    "redirects": 1,
    "exsectionformat": "plain"
}   

    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "PythonRAG/1.0"},
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()
    

    pages = data["query"]["pages"]
    # print(f"Fetched {len(pages)} pages for title '{title}'")
    page = next(iter(pages.values()))
    # print(page)
    # pages = data["query"]["pages"]

    text = page["extract"]

    return text

if __name__ == "__main__":
    text = fetch_wikipedia("Transformer (machine learning model)")
    
    print(f"Total characters: {len(text)}")
    print(f"Total words:      {len(text.split())}")
    print(f"\nFirst 1000 chars:\n{text[:1000]}")