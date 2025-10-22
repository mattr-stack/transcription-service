"""
transcribe_with_speakers_Final.py - Improved Version

Usage:
    python transcribe_with_speakers_Final.py <airtable_record_id> [--local-audio PATH] [--test-seconds N]

Key env vars (required):
    AIRTABLE_API_KEY         # Airtable API key
    HF_TOKEN                 # HuggingFace token (needed for diarization)
    
Key env vars (optional):
    AIRTABLE_BASE_ID, AIRTABLE_TABLE, STATUS_FIELD,
    TRANSCRIPT_URL_FIELD, AUDIO_LINK_FIELD, NAME_FIELD

    GOOGLE_OAUTH_CREDS (default credentials.json)
    GOOGLE_TOKEN_JSON  (default token.json)

    FORCE_CPU=1              # force CPU
    WHISPERX_MODEL=base      # model size (default large-v2)
    DO_DIARIZATION=0/1       # default 1
    SLACK_WEBHOOK_URL        # Slack notifications

    USE_CACHE=1/0            # default 1
    REFRESH_CACHE=1/0        # default 0
    PAUSE_SEC=1.2            # merge threshold (seconds)
    DOC_HTTP_TIMEOUT=180     # Google Docs HTTP timeout
    DOC_CHUNK_SIZE=20000     # chars per Docs batchUpdate
"""
import warnings

# ─── Suppress specific startup warnings ─────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message=r"Module 'speechbrain\\.pretrained' was deprecated, redirecting to 'speechbrain\\.inference'.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=r".*degrees of freedom is <= 0.*",
    category=UserWarning
)
# ────────────────────────────────────────────────────────────────────────────────

# ---- CUDA DLL SAFETY NET (must be before importing torch/whisperx) ----
import os, sys
from pathlib import Path

def _add_cuda_dll_dirs():
    if not hasattr(os, "add_dll_directory"):
        return
    venv_root = Path(sys.executable).parent.parent  # ...\whisperx-env
    sp = venv_root / "Lib" / "site-packages" / "nvidia"
    for d in [
        sp / "cudnn" / "bin",
        sp / "cublas" / "bin",
        sp / "cuda_nvrtc" / "bin",
        sp / "cuda_runtime" / "bin",
        venv_root / "Lib" / "site-packages" / "torch" / "lib",
    ]:
        if d.exists():
            os.add_dll_directory(str(d))

_add_cuda_dll_dirs()
# ----------------------------------------------------------------------

import io, re, tempfile, subprocess, logging, time, argparse, shutil, atexit, signal, gc
from typing import Optional
from hashlib import sha1
import requests, json
import psutil

# ─── Speaker‐name extraction & assignment ─────────────────────────────────
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_names(text: str) -> list[str]:
    names = set()
    # explicit “this is Name” patterns
    for m in re.finditer(r"\bthis is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", text):
        names.add(m.group(1))
    # spaCy PERSON entities
    for ent in nlp(text).ents:
        if ent.label_ == "PERSON":
            names.add(ent.text)
    return list(names)

def assign_names(transcript: str, names: list[str]) -> str:
    speaker_map = {}
    out_lines = []
    for line in transcript.splitlines():
        # 1) split timestamp from rest
        ts_part, sep, rest = line.partition("]")
        timestamp = ts_part + sep  # e.g. "[00:00:00]"
        rest = rest.strip()        # e.g. "Speaker 3: You guys love…"

        # 2) pull off the original label and the actual text
        orig_label, sep2, text = rest.partition(":")
        spk_num = re.search(r"Speaker\s+(\d+)", orig_label)
        label = spk_num.group(1) if spk_num else None

        # 3) on first mention, map to a real name
        if label and label not in speaker_map:
            for name in names:
                if name in text:
                    speaker_map[label] = name
                    break

        # 4) fallback to "Speaker N" if unmapped
        label_name = speaker_map.get(label, f"Speaker {label}" if label else orig_label)

        # 5) reassemble
        out_lines.append(f"{timestamp} {label_name}: {text.strip()}")

    return "\n".join(out_lines)
# ─────────────────────────────────────────────────────────────────────────────

# ─── CONFIGURATION VALIDATION AND SETUP ─────────────────────────────────────────
def validate_required_env_vars():
    """Validate required environment variables."""
    required_vars = ["AIRTABLE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}", file=sys.stderr)
        print("Please set the following environment variables:", file=sys.stderr)
        for var in missing:
            print(f"  {var}", file=sys.stderr)
        return False
    
    return True

# Validate required vars before proceeding
if not validate_required_env_vars():
    sys.exit(1)

# -------- CONFIG (env vars override these) --------
AIRTABLE_API_KEY    = os.getenv("AIRTABLE_API_KEY")
BASE_ID             = os.getenv("AIRTABLE_BASE_ID", "apprqVh4x9vqtzbAC")
TABLE_NAME          = os.getenv("AIRTABLE_TABLE", "All Content")
STATUS_FIELD        = os.getenv("STATUS_FIELD", "Transcription Status")
TRANSCRIPT_URL_FLD  = os.getenv("TRANSCRIPT_URL_FIELD", "Transcript URL")
AUDIO_LINK_FIELD    = os.getenv("AUDIO_LINK_FIELD", "Link to Article Audio")
NAME_FIELD          = os.getenv("NAME_FIELD", "Name")
DRIVE_FOLDER_ID     = os.getenv("DRIVE_FOLDER_ID", "1DDXGw9CRSVrykAVuEuijl6DnXVBJZR_k")

CREDENTIALS_JSON    = os.getenv("GOOGLE_OAUTH_CREDS", "credentials.json")
TOKEN_JSON          = os.getenv("GOOGLE_TOKEN_JSON", "token.json")

DEVICE              = "cuda" if os.getenv("FORCE_CPU", "0") != "1" else "cpu"
WHISPERX_MODEL      = os.getenv("WHISPERX_MODEL", "large-v2")
DO_DIARIZATION      = os.getenv("DO_DIARIZATION", "1") == "1"
HF_TOKEN            = os.getenv("HF_TOKEN")

PAUSE_SEC           = float(os.getenv("PAUSE_SEC", "1.2"))
SLACK_WEBHOOK_URL   = os.getenv("SLACK_WEBHOOK_URL")

AUDIO_OK            = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/documents"
]

USE_CACHE     = os.getenv("USE_CACHE", "1") == "1"
REFRESH_CACHE = os.getenv("REFRESH_CACHE", "0") == "1"
CACHE_DIR     = Path.home() / ".whisperx_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------- LOGGING SETUP ----------
def setup_logging(record_id: str = None):
    """Setup logging with optional record-specific context."""
    log_format = "%(asctime)s %(levelname)s"
    if record_id:
        log_format += f" [REC:{record_id[:8]}]"
    log_format += " %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

setup_logging()
log = logging.getLogger("transcriber")

# ---------- RESOURCE MANAGEMENT ----------
def cleanup():
    """Cleanup resources on exit."""
    log.info("Cleaning up resources...")
    gc.collect()

def signal_handler(signum, frame):
    log.info("Received signal %s, exiting gracefully", signum)
    cleanup()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_memory_usage():
    """Check and log memory usage."""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        log.info(f"Memory usage: {memory_mb:.1f} MB")
        
        if memory_mb > 8000:  # 8GB threshold
            log.warning("High memory usage detected, triggering garbage collection")
            gc.collect()
    except Exception as e:
        log.warning(f"Could not check memory usage: {e}")

# ---------- THIRD-PARTY ----------
from pyairtable import Table
import whisperx

GOOGLE_OK = True
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.exceptions import RefreshError
    from google_auth_httplib2 import AuthorizedHttp
    import httplib2
except Exception as e:
    GOOGLE_OK = False
    log.warning("Google API libs missing: %s", e)

# ─── CONFIGURATION VALIDATION ─────────────────────────────────────────────────
def validate_config():
    """Validate all configuration settings."""
    errors = []
    warnings = []
    
    if not AIRTABLE_API_KEY:
        errors.append("AIRTABLE_API_KEY is required")
    
    if GOOGLE_OK and not Path(CREDENTIALS_JSON).exists():
        warnings.append(f"Google credentials file not found: {CREDENTIALS_JSON} (Google Drive features disabled)")
    
    if DO_DIARIZATION and not HF_TOKEN:
        errors.append("HF_TOKEN required when DO_DIARIZATION=1")
    
    try:
        import torch
        global DEVICE
        if DEVICE == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available, falling back to CPU")
            DEVICE = "cpu"
    except ImportError:
        errors.append("PyTorch not installed")
    
    if SLACK_WEBHOOK_URL and not SLACK_WEBHOOK_URL.startswith("https://hooks.slack.com"):
        warnings.append("SLACK_WEBHOOK_URL doesn't look like a valid Slack webhook")
    
    # Log warnings
    for warning in warnings:
        log.warning("Config warning: %s", warning)
    
    # Log and return errors
    if errors:
        for error in errors:
            log.error("Config error: %s", error)
        return False
    
    return True

# Validate configuration
if not validate_config():
    sys.exit(1)

# ─── SLACK ERROR ALERTS ─────────────────────────────────────────────────────────
def send_slack_error(record_id: str, error_message: str):
    """Send error alert to Slack if webhook is configured."""
    if not SLACK_WEBHOOK_URL:
        return
        
    payload = {
        "text": (
            ":rotating_light: *Transcription Error Alert*\n"
            f"• *Record ID:* `{record_id}`\n"
            f"• *Error:* `{error_message}`"
        )
    }
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
        if resp.status_code == 200:
            log.info("Slack alert sent.")
        else:
            log.warning(f"Slack alert failed: {resp.status_code}")
    except Exception as slack_err:
        log.warning(f"Failed to send Slack alert: {slack_err}")

def send_slack_success(record_id: str, doc_url: str):
    """Send success notification to Slack."""
    if not SLACK_WEBHOOK_URL:
        return
        
    payload = {
        "text": (
            ":white_check_mark: *Transcription Complete*\n"
            f"• *Record ID:* `{record_id}`\n"
            f"• *Document:* {doc_url}"
        )
    }
    try:
        requests.post(
            SLACK_WEBHOOK_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
    except Exception:
        pass  # Don't fail the main process for success notifications

# ────────────────────────────────────────────────────────────────────────────

def update_progress(table, record_id, status, progress_pct=None):
    """Update record status with optional progress percentage."""
    update_data = {STATUS_FIELD: status}
    if progress_pct is not None:
        # Only add progress if the field exists in your Airtable
        # Remove this line if you don't have a Progress field
        # update_data["Progress"] = f"{progress_pct}%"
        pass
    
    try:
        table.update(record_id, update_data)
        log.info(f"Updated status to: {status}")
    except Exception as e:
        log.warning("Failed to update progress: %s", e)

# ---------- HELPERS ----------
def slugify(txt: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', txt).strip('_')

def extract_drive_id(url: str) -> Optional[str]:
    m = re.search(r"/file/d/([^/]+)/", url)
    return m.group(1) if m else None

def cache_key(url: str, fid: Optional[str]) -> str:
    if fid:
        return f"drive_{fid}"
    return "url_" + sha1(url.encode("utf-8")).hexdigest()[:16]

def get_creds() -> Optional["Credentials"]:
    if not GOOGLE_OK:
        return None
    creds = None
    if os.path.exists(TOKEN_JSON):
        creds = Credentials.from_authorized_user_file(TOKEN_JSON, SCOPES)
    try:
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise RefreshError("No valid refresh token")
    except RefreshError:
        if not os.path.exists(CREDENTIALS_JSON):
            log.error("Google credentials file not found: %s", CREDENTIALS_JSON)
            return None
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_JSON, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_JSON, "w") as token:
            token.write(creds.to_json())
    return creds

def drive_download(file_id: str, dest: Path, creds: "Credentials"):
    http = httplib2.Http(timeout=int(os.getenv("DOC_HTTP_TIMEOUT", "180")))
    authed_http = AuthorizedHttp(creds, http=http)
    drive = build("drive", "v3", http=authed_http, cache_discovery=False)

    request = drive.files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=25 * 1024 * 1024)
        done = False
        last_progress = -1.0
        stall_check_ts = time.time()
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = status.progress() * 100
                if pct - last_progress >= 1.0:
                    log.info("Download %.1f%%", pct)
                    last_progress = pct
                    stall_check_ts = time.time()
            if time.time() - stall_check_ts > 180:
                raise TimeoutError("Drive download stalled")

def http_download(url: str, dest: Path):
    import requests
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

def is_probably_html(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return False

def download_any(url: str, tmp_dest: Path, creds: Optional["Credentials"]) -> Path:
    fid = extract_drive_id(url)
    key = cache_key(url, fid)
    cache_path = CACHE_DIR / key

    if USE_CACHE and cache_path.exists() and cache_path.stat().st_size > 0 and not REFRESH_CACHE:
        log.info("Using cached file: %s (%.1f MB)", cache_path.name, cache_path.stat().st_size / (1024 * 1024))
        return cache_path

    log.info("Cache miss or refresh. Downloading...")
    try:
        if fid and creds:
            drive_download(fid, tmp_dest, creds)
        else:
            http_download(url, tmp_dest)

        if tmp_dest.stat().st_size < 1024 or is_probably_html(tmp_dest):
            raise RuntimeError("Downloaded file looks invalid (HTML or tiny).")

        try:
            tmp_dest.replace(cache_path)
        except Exception:
            shutil.copy2(tmp_dest, cache_path)

        return cache_path
    except Exception as e:
        log.error(f"Download failed: {e}")
        raise

def prepare_audio(src: Path, name_field: str, record_id: str, test_seconds: int = 0) -> Path:
    base = f"{slugify(name_field)}_{record_id[:6]}"
    ext  = src.suffix.lower()

    if ext in AUDIO_OK and test_seconds == 0:
        dest = src.with_name(base + ext)
        if dest != src:
            shutil.copy2(src, dest)
        log.info("Audio OK (%s). Using %s", ext, dest.name)
        return dest

    dest = src.with_name(base + ".wav")
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if test_seconds > 0:
        cmd += ["-t", str(test_seconds)]
    cmd += ["-ar", "16000", "-ac", "1", str(dest)]
    log.info("ffmpeg: %s", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return dest
    except subprocess.CalledProcessError as e:
        log.error(f"ffmpeg failed: {e}")
        log.error(f"ffmpeg stderr: {e.stderr}")
        raise

# ---- Diarizer helper (handles different whisperx versions) ----
def get_diarizer(device: str):
    DP = None
    try:
        from whisperx.diarize import DiarizationPipeline as DP  # newer path
    except Exception:
        DP = getattr(whisperx, "DiarizationPipeline", None)      # older path
    if DP is None:
        return None
    return DP(use_auth_token=HF_TOKEN, device=device)

def run_whisperx(input_audio: Path, table, record_id: str) -> dict:
    log.info("Loading WhisperX model (%s)", WHISPERX_MODEL)
    update_progress(table, record_id, "Loading Model", 10)
    
    try:
        model = whisperx.load_model(WHISPERX_MODEL, device=DEVICE, language="en")
        check_memory_usage()
    except Exception as e:
        log.error(f"Failed to load WhisperX model: {e}")
        raise

    log.info("Transcribing...")
    update_progress(table, record_id, "Transcribing", 25)
    try:
        result = model.transcribe(str(input_audio))
        check_memory_usage()
    except Exception as e:
        log.error(f"Transcription failed: {e}")
        raise

    log.info("Aligning words...")
    update_progress(table, record_id, "Aligning", 50)
    try:
        model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, str(input_audio), DEVICE)
        check_memory_usage()
    except Exception as e:
        log.error(f"Alignment failed: {e}")
        raise

    if DO_DIARIZATION and HF_TOKEN:
        log.info("Running diarization...")
        update_progress(table, record_id, "Diarizing", 75)
        try:
            diarizer = get_diarizer(DEVICE)
            if diarizer is None:
                log.warning("No DiarizationPipeline found in whisperx. Skipping diarization.")
            else:
                diarize_segments = diarizer(str(input_audio))
                result = whisperx.assign_word_speakers(diarize_segments, result)
                check_memory_usage()
        except Exception as e:
            log.error(f"Diarization failed: {e}")
            # Don't fail the entire process for diarization errors
            log.warning("Continuing without diarization")
    else:
        log.info("Diarization skipped")

    update_progress(table, record_id, "Formatting", 90)
    return result

def normalize_speakers(result: dict) -> dict:
    segs = result.get("segments", [])
    real_tags = {s.get("speaker") for s in segs if s.get("speaker") and s.get("speaker") != "Speaker?"}
    if not real_tags:
        for s in segs:
            s["speaker"] = "Speaker 1"
        return result

    mapping, next_id = {}, 1
    for s in segs:
        raw = s.get("speaker")
        if not raw or raw == "Speaker?":
            raw = "UNK"
        if raw not in mapping:
            mapping[raw] = f"Speaker {next_id}"
            next_id += 1
        s["speaker"] = mapping[raw]
    return result

def format_transcript(result: dict, pause_sec: float = 1.2) -> str:
    segs = result.get("segments", [])
    merged = []
    cur = None

    for s in segs:
        spk = s.get("speaker") or (cur["speaker"] if cur else "Speaker 1")
        start, end = s.get("start", 0.0), s.get("end", 0.0)
        text = s.get("text", "").strip()

        if cur and cur["speaker"] == spk and (start - cur["end"]) <= pause_sec:
            cur["text"] += (" " if cur["text"] else "") + text
            cur["end"] = end
        else:
            cur = {"speaker": spk, "start": start, "end": end, "text": text}
            merged.append(cur)

    lines = []
    for m in merged:
        ts = f"{int(m['start']//3600):02d}:{int((m['start']%3600)//60):02d}:{int(m['start']%60):02d}"
        lines.append(f"[{ts}] {m['speaker']}: {m['text']}")
    return "\n".join(lines)

def google_upload_doc(title: str, text_path: Path, creds: Optional["Credentials"]) -> Optional[str]:
    if not creds:
        log.warning("No Google credentials available, cannot upload to Google Docs")
        return None

    timeout = int(os.getenv("DOC_HTTP_TIMEOUT", "180"))
    chunk_size = int(os.getenv("DOC_CHUNK_SIZE", "20000"))

    try:
        # Prepare authenticated HTTP
        raw_http = httplib2.Http(timeout=timeout)
        authed_http = AuthorizedHttp(creds, http=raw_http)

        # Build Docs & Drive services
        docs  = build("docs",  "v1", http=authed_http, cache_discovery=False)
        drive = build("drive", "v3", http=authed_http, cache_discovery=False)

        # 1) Create a new Google Doc
        body = {"title": title}
        doc = docs.documents().create(body=body).execute(num_retries=3)
        doc_id = doc["documentId"]

        # 2) Move it into your target folder
        try:
            drive.files().update(
                fileId=doc_id,
                addParents=DRIVE_FOLDER_ID,
                removeParents=None,
                fields="id, parents"
            ).execute(num_retries=3)
        except Exception as e:
            log.warning("Could not move Doc into folder %s: %s", DRIVE_FOLDER_ID, e)

        # 3) Batch‐upload the transcript text
        content = text_path.read_text(encoding="utf-8")
        chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
        for i, chunk in enumerate(chunks):
            log.info(f"Uploading chunk {i+1}/{len(chunks)} to Google Doc")
            req = [{
                "insertText": {
                    "endOfSegmentLocation": {},
                    "text": chunk
                }
            }]
            docs.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": req}
            ).execute(num_retries=3)

        return f"https://docs.google.com/document/d/{doc_id}/edit"

    except Exception as e:
        log.error("Google Docs upload failed: %s", e)

        # Fallback: upload as plain .txt into the same folder
        try:
            log.info("Attempting fallback upload to Google Drive as text file")
            media = MediaFileUpload(str(text_path), mimetype="text/plain", resumable=True)
            f = drive.files().create(
                body={
                    "name": title + ".txt",
                    "parents": [DRIVE_FOLDER_ID]
                },
                media_body=media,
                fields="id, parents"
            ).execute(num_retries=3)
            return f"https://drive.google.com/file/d/{f['id']}/view"
        except Exception as e2:
            log.error("Drive upload also failed: %s", e2)
            return None

# ---------- MAIN ----------
def main(args) -> int:
    record_id    = args.record_id
    local_audio  = args.local_audio
    test_seconds = args.test_seconds

    # Setup logging with record ID context
    setup_logging(record_id)
    
    log.info("Starting transcription for record %s", record_id)
    
    try:
        table = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
    except Exception as e:
        log.error("Failed to connect to Airtable: %s", e)
        return 1

    try:
        rec = table.get(record_id)
    except Exception as e:
        log.error("Failed to fetch Airtable record: %s", e)
        return 1

    try:
        update_progress(table, record_id, "Processing", 5)
    except Exception as e:
        log.warning("Could not set status to Processing: %s", e)

    name_field = rec["fields"].get(NAME_FIELD, f"rec_{record_id[:6]}")
    url        = rec["fields"].get(AUDIO_LINK_FIELD)

    log.info(f"Processing: {name_field}")

    creds = get_creds() if GOOGLE_OK else None
    if not creds:
        log.warning("No Google credentials available - Google Drive/Docs features disabled")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            if local_audio:
                log.info("Using local file: %s", local_audio)
                src_path = Path(local_audio).resolve()
                if not src_path.exists():
                    raise FileNotFoundError(f"Local file not found: {local_audio}")
            else:
                if not url:
                    raise ValueError(f"No URL in field '{AUDIO_LINK_FIELD}'")

                temp_download = tmpdir / "downloaded_file"
                log.info("Downloading %s", url)
                src_path = download_any(url, temp_download, creds)

                size = src_path.stat().st_size if src_path.exists() else 0
                log.info("File size: %.1f MB", size / (1024 * 1024))

            try:
                audio_path = prepare_audio(src_path, name_field, record_id, test_seconds)
            except Exception as e:
                raise RuntimeError(f"Audio preparation failed: {e}")

            try:
                result = run_whisperx(audio_path, table, record_id)
            except Exception as e:
                raise RuntimeError(f"WhisperX processing failed: {e}")

            result = normalize_speakers(result)
            # format the raw transcript
            raw = format_transcript(result, pause_sec=PAUSE_SEC)
            # extract and assign real speaker names
            names = extract_names(raw)
            transcript_text = assign_names(raw, names)
            txt_path = tmpdir / (audio_path.stem + ".txt")
            txt_path.write_text(transcript_text, encoding="utf-8")

            log.info("Uploading transcript...")
            doc_url = google_upload_doc(name_field, txt_path, creds)
            if not doc_url:
                doc_url = f"file:///{txt_path}"
                log.warning("Could not upload to Google - saving local path")

            try:
                table.update(record_id, {
                    STATUS_FIELD: "Transcribed",
                    TRANSCRIPT_URL_FLD: doc_url
                })
                log.info("Successfully updated Airtable record")
                send_slack_success(record_id, doc_url)
            except Exception as e:
                log.warning("Airtable update failed: %s", e)

        log.info("Transcription completed successfully for record %s", record_id)
        return 0

    except Exception as e:
        error_msg = str(e)
        log.error("Transcription failed for %s: %s", record_id, error_msg)
        
        # Send Slack alert
        send_slack_error(record_id, error_msg)
        
        # Update Airtable status
        try:
            table.update(record_id, {STATUS_FIELD: "Error"})
        except Exception as update_err:
            log.error("Failed to update status to Error: %s", update_err)
        
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    parser.add_argument("record_id", help="Airtable record ID")
    parser.add_argument("--local-audio", help="Path to local audio/video file to skip download")
    parser.add_argument("--test-seconds", type=int, default=0, help="Trim to first N seconds for quick tests")
    args = parser.parse_args()
    
    log.info("=== Transcription Script Started ===")
    log.info(f"Record ID: {args.record_id}")
    log.info(f"Device: {DEVICE}")
    log.info(f"Model: {WHISPERX_MODEL}")
    log.info(f"Diarization: {DO_DIARIZATION}")
    
    sys.exit(main(args))
