import dataclasses
import os
import secrets
import time
from collections import deque
import warnings
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from pathlib import Path
from typing import List

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


# --- Initialization ---
UPLOAD_DIR.mkdir(exist_ok=True)

TOKEN_BUFFER = deque(maxlen=N_MAX_VALID_TOKENS)
"""
FIFO buffer for tokens stored as tuples of
`(token, expiry_unix_timestamp)`
"""

# --- Backend functionalities ---


@dataclasses.dataclass
class OcrDataFrame:
    input_filename: str
    local_path: Path
    markdown_text: str = ""


def batch_process_ocr(
    input_dataframes: list[OcrDataFrame], to_cleanup=True
) -> list[OcrDataFrame]:
    """Running OCR on multiple files and return updated results."""

    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    
    output_dataframes: list[OcrDataFrame] = []
    try:
        all_results: list[dict] = ocr_engine.predict(
            [str(df.local_path) for df in input_dataframes]
        )
        for result in all_results:
            df = next(
                (df for df in input_dataframes if str(df.local_path) == result["input_path"]),
                None,
            )

            if df is None:
                warnings.warn(f"Possibly corrupted OCR result: {result}")
                continue

            df.markdown_text = "\n".join(result["rec_texts"])
            output_dataframes.append(df)

    except Exception as e:
        warnings.warn(f"Failed OCR with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if to_cleanup:
            for df in input_dataframes:
                df.local_path.unlink(missing_ok=True)

    return output_dataframes


# --- Endpoints ---


@app.get("/api/get-token")
async def get_token():
    """Generate a timed token and push to the buffer."""
    new_token = secrets.token_urlsafe(32)
    expiry = time.time() + TOKEN_EXPIRY_SEC

    TOKEN_BUFFER.append({"token": new_token, "expires": expiry})

    return {"token": new_token}


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


@app.post("/api/ocr")
async def run_ocr(
    files: List[UploadFile] = File(...), authenticated: bool = Depends(verify_token)
):
    """Accept files, verify token and trigger OCR."""

    if not authenticated:
        raise HTTPException(status_code=401, detail="Failed authentication.")

    # store files
    ocr_dataframes: list[OcrDataFrame] = []
    for file in files:
        assert file.filename is not None

        local_path = UPLOAD_DIR / f"{secrets.token_hex(8)}_{file.filename}"

        with open(local_path, "wb") as f:
            content = await file.read()
            f.write(content)

        ocr_dataframes.append(
            OcrDataFrame(input_filename=file.filename, local_path=local_path)
        )

    # trigger OCR
    ocr_dataframes = batch_process_ocr(ocr_dataframes)

    return {
        "results": [
            dict(
                input_filename=df.input_filename,
                markdown_text=df.markdown_text,
            )
            for df in ocr_dataframes
        ]
    }


# host the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
