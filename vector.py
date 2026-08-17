from pathlib import Path
import pickle

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: list[str] = []

    @property
    def count(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, chunks: list[str]) -> None:
        self.index.add(np.asarray(embeddings, dtype="float32"))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 4) -> list[str]:
        if not self.count:
            return []
        _, indices = self.index.search(
            np.asarray([query_embedding], dtype="float32"), min(k, self.count)
        )
        return [self.chunks[i] for i in indices[0] if i != -1]

    def save(self, path: str | Path) -> None:
        base_path = Path(path)
        faiss.write_index(self.index, str(base_path.with_suffix(".index")))
        with base_path.with_suffix(".chunks.pkl").open("wb") as file:
            pickle.dump(self.chunks, file)

    def load(self, path: str | Path) -> bool:
        base_path = Path(path)
        index_path = base_path.with_suffix(".index")
        chunks_path = base_path.with_suffix(".chunks.pkl")
        if not index_path.exists() or not chunks_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with chunks_path.open("rb") as file:
            self.chunks = pickle.load(file)
        return True
