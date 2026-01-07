import asyncio
import dataclasses
import os
import secrets
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from http import HTTPStatus
from paddleocr import PaddleOCR
from pathlib import Path
from typing import List
from uuid import UUID, uuid4

app = FastAPI()

# allow cross-origin requests in development environment
# this is useful when running "npm run dev"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Configuration ---
UPLOAD_DIR = Path("/tmp/paddle-uploads")

N_MAX_VALID_TOKENS = 20
"""
The maximum number of concurrently valid tokens.
This controls client session number.
"""

TOKEN_EXPIRY_SEC = 300
"""Token expiry time in seconds"""

# load hosting IP and port from the environment variables
HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", "8010"))


# --- Definitions ---


@dataclasses.dataclass
class Job:
    """Reference: https://stackoverflow.com/a/63171013/27092911"""

    uid: UUID = dataclasses.field(default_factory=uuid4)
    status: str = "processing"
    messages: str = "Already started."
    progress: float = 0.0
    results: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    """Results as a `list` of `(remote_filename, markdown_text)` as a `str` tuple."""


# --- Initialization ---
UPLOAD_DIR.mkdir(exist_ok=True)

TOKEN_BUFFER = deque(maxlen=N_MAX_VALID_TOKENS)
"""
FIFO buffer for tokens stored as tuples of
`(token, expiry_unix_timestamp)`
"""

jobs: dict[UUID, Job] = {}


# --- Backend functionalities ---


async def verify_token(authorization: str = Header(...)):
    """Middleware-style check for token validity."""
    token = authorization.replace("Bearer ", "")
    current_time = time.time()

    # check if token exists in buffer and hasn't expired
    valid_token = next((t for t in TOKEN_BUFFER if t["token"] == token), None)

    if not valid_token:
        raise HTTPException(status_code=401, detail="Invalid or rotated token")
    if current_time > valid_token["expires"]:
        raise HTTPException(status_code=401, detail="Token expired")

    return True


def _convert_image_to_text(ocr_engine: PaddleOCR, local_path: Path | str) -> str:
    """Running OCR on multiple files and return updated results."""

    try:
        results: list[dict] = ocr_engine.predict(str(local_path))
    except Exception as e:
        warnings.warn(f'Failed OCR with file "{local_path}" and error:\n{e}')
        # TODO: resize the image to 2000x2000 or below,
        # and retry OCR in the backend automatically
        return "(Failed. Please reduce the image size.)"

    if len(results) == 0:
        warnings.warn(f"Empty result for input image: {local_path}")
        return ""

    if len(results) > 1:
        warnings.warn(f"Possibly corrupted OCR results:\n{results}")

    return "\n".join(results[0]["rec_texts"])


async def convert_multiple_images_to_text(
    file_mappings: list[tuple[str, Path]], uid: UUID
):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()
    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=False,
    )

    # trigger OCR individually to avoid no result from only
    # a single failed file
    single_progress_step = 1 / len(file_mappings)
    for remote_filename, local_path in file_mappings:
        jobs[uid].messages = f"Processing {remote_filename}."
        markdown_text = await loop.run_in_executor(
            executor,
            _convert_image_to_text,
            ocr_engine,
            local_path,
        )
        jobs[uid].progress += single_progress_step
        jobs[uid].results.append((remote_filename, markdown_text))

    executor.shutdown()

    # mark task as completed
    jobs[uid].progress = 1.0
    jobs[uid].messages = "Completed."
    jobs[uid].status = "completed"

    # clean up
    for _, local_path in file_mappings:
        local_path.unlink(missing_ok=True)


# --- Endpoints ---


@app.get("/api/get-token")
async def get_token():
    """Generate a timed token and push to the buffer."""
    new_token = secrets.token_urlsafe(32)
    expiry = time.time() + TOKEN_EXPIRY_SEC

    TOKEN_BUFFER.append({"token": new_token, "expires": expiry})

    return {"token": new_token}


@app.post("/api/ocr", status_code=HTTPStatus.ACCEPTED)
async def run_ocr(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    authenticated: bool = Depends(verify_token),
):
    """Accept files, verify token and trigger OCR."""

    if not authenticated:
        raise HTTPException(status_code=401, detail="Failed authentication.")

    # store files
    file_mappings: list[tuple[str, Path]] = []
    for file in files:
        assert file.filename is not None, f"Invalid file: {file}"

        local_path: Path = UPLOAD_DIR / f"{secrets.token_hex(8)}_{file.filename}"
        with open(local_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_mappings.append((file.filename, local_path))

    # trigger OCR
    new_task = Job()
    jobs[new_task.uid] = new_task
    background_tasks.add_task(
        convert_multiple_images_to_text, file_mappings, new_task.uid
    )

    return {"task_uid": new_task.uid}


@app.get("/api/get-status/{uid}")
async def get_status(uid: UUID):
    task = jobs.get(uid, None)
    if task is None:
        raise HTTPException(HTTPStatus.NOT_FOUND)

    return dict(
        status=task.status,
        progress=task.progress,
        messages=task.messages,
    )


@app.get("/api/get-results/{uid}")
async def get_results(uid: UUID):
    task = jobs.get(uid, None)
    if task is None:
        raise HTTPException(HTTPStatus.NOT_FOUND)

    results = task.results
    del jobs[uid]

    return dict(results=results)


# host the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
