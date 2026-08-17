from embed import embd_chunks, embd_query
from load_pdf import loader
from vector import VectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os
store = VectorStore(dim=384)
store.load("my_index")


def answer_query(query : str, k : int = 4):
    load_dotenv()

    api_key = os.getenv("OPEN_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    client = OpenAI(api_key=api_key)
    
    
    query_vec = embd_query(query)
    retrieved_chunks = store.search(query_vec, k = k)
    
    context = " ---\n \n \n ---".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

context : {context}

question : {query}"""

    response = client.messages.create(
        model = "model",
        max_tokens = 500,
        messages = [{"role" : "user","content" : prompt}]
    )
    return response.output_text