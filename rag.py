import os
from pathlib import Path

from embed import embd_chunks, embd_query
from load_pdf import chunking, loader
from vector import VectorStore


BASE_DIRECTORY = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIRECTORY / "my_index"
store = VectorStore(dim=384)
store.load(INDEX_PATH)


def index_files(file_paths: list[Path]) -> int:
    all_chunks = []
    for file_path in file_paths:
        text = loader(file_path)
        chunks = chunking(text)
        all_chunks.extend(f"Source: {file_path.name}\n{chunk}" for chunk in chunks if chunk)

    if not all_chunks:
        return 0

    store.add(embd_chunks(all_chunks), all_chunks)
    store.save(INDEX_PATH)
    return len(all_chunks)


def answer_query(query: str, k: int = 4) -> str:
    if not store.count:
        return "No document content has been indexed yet. Upload files first."

    retrieved_chunks = store.search(embd_query(query), k=k)
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say that you do not know.

Context:
{context}

Question: {query}"""

    api_key = os.getenv("api_key")
    model = os.getenv("model")
    if not api_key or not model:
        return "The RAG index is ready, but ANTHROPIC_API_KEY and ANTHROPIC_MODEL are not configured."

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
