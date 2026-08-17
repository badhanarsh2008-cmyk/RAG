from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from embed import embd_chunks
from load_pdf import chunking, loader
from vector import VectorStore



BASE_DIRECTORY = Path(__file__).resolve().parent
UPLOAD_DIRECTORY = BASE_DIRECTORY / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIRECTORY / "static"), name="static")


class Query(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIRECTORY / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return (BASE_DIRECTORY / "templates" / "chat-room.html").read_text(encoding="utf-8")


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    UPLOAD_DIRECTORY.mkdir(exist_ok=True)
    saved_paths = []
    saved_names = []

    for uploaded_file in files:
        original_name = Path(uploaded_file.filename or "").name
        suffix = Path(original_name).suffix.lower()
        if not original_name or suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only PDF, TXT, and DOCX files are allowed.")

        destination = UPLOAD_DIRECTORY / f"{uuid4().hex}{suffix}"
        with destination.open("wb") as output_file:
            while content := await uploaded_file.read(1024 * 1024):
                output_file.write(content)
        await uploaded_file.close()
        saved_paths.append(destination)
        saved_names.append(original_name)

    try:
        store = VectorStore(dim=384)
        store.load(BASE_DIRECTORY / "my_index")
        all_chunks = []

        for file_path in saved_paths:
            all_chunks.extend(chunking(loader(file_path)))

        if all_chunks:
            store.add(embd_chunks(all_chunks), all_chunks)
            store.save(BASE_DIRECTORY / "my_index")
        chunk_count = len(all_chunks)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Files were saved, but indexing failed: {error}",
        ) from error

    return {
        "message": f"Saved and indexed {len(saved_names)} file(s) into {chunk_count} chunks.",
        "files": saved_names,
    }


@app.post("/ask")
def ask(query: Query):
    from rag import answer_query

    return {"answer": answer_query(query.question)}
