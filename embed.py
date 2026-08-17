from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


def embd_chunks(chunks: list[str]):
    return get_model().encode(chunks, show_progress_bar=False, convert_to_numpy=True)


def embd_query(query: str):
    return get_model().encode([query], convert_to_numpy=True)[0]
