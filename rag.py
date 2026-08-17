from pathlib import Path

from embed import embd_query
from vector import VectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

BASE_DIRECTORY = Path(__file__).resolve().parent
store = VectorStore(dim=384)
store.load(BASE_DIRECTORY / "my_index")


def answer_query(query : str, k : int = 4):
    load_dotenv(BASE_DIRECTORY / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1")
    
    
    query_vec = embd_query(query)
    retrieved_chunks = store.search(query_vec, k = k)
    
    context = " ---\n \n \n ---".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

context : {context}

question : {query}"""

    response = client.responses.create(model=model, input=prompt)
    return response.output_text
