# app.py - COMPLETE AND MERGED VERSION
import warnings

# Suppress specific startup warnings
warnings.filterwarnings(
    "ignore",
    message=r"Module 'speechbrain\.pretrained' was deprecated, redirecting to 'speechbrain\.inference'.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=r".*degrees of freedom is <= 0.*",
    category=UserWarning
)

import os, sys, io, re, tempfile, subprocess, logging, time, shutil, gc
from pathlib import Path
from typing import Optional
from hashlib import sha1

# Third-party imports
import requests, json
import psutil
import spacy
from flask import Flask, request, jsonify
from pyairtable import Table
import whisperx

# Imports for Google API
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.exceptions import RefreshError
    from google_auth_httplib2 import AuthorizedHttp
    import httplib2
    GOOGLE_OK = True
except Exception as e:
    GOOGLE_OK = False
    logging.warning("Google API libs missing: %s", e)
# Verify GPU setup on startup
import torch
logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"CUDA device: {torch.cuda.get_device_name()}")
    logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# ─── spaCy Model Initialization ─────────────────────────────────
nlp = spacy.load("en_core_web_sm")

# ─── CONFIGURATION (from env vars) ─────────────────────────────────────────
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

# ─── LOGGING SETUP ─────────────────────────────────────────
def setup_logging(record_id: str = None):
    log_format = "%(asctime)s %(levelname)s"
    if record_id:
        log_format += f" [REC:{record_id[:8]}]"
    log_format += " %(message)s"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Initialize logging for the app startup
setup_logging()

# ─── HELPER FUNCTIONS (from original script) ─────────────────────────────────
def extract_names(text: str) -> list[str]:
    names = set()
    for m in re.finditer(r"\bthis is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", text):
        names.add(m.group(1))
    for ent in nlp(text).ents:
        if ent.label_ == "PERSON":
            names.add(ent.text)
    return list(names)

def assign_names(transcript: str, names: list[str]) -> str:
    speaker_map = {}
    out_lines = []
    for line in transcript.splitlines():
        ts_part, sep, rest = line.partition("]")
        timestamp = ts_part + sep
        rest = rest.strip()
        orig_label, sep2, text = rest.partition(":")
        spk_num = re.search(r"Speaker\s+(\d+)", orig_label)
        label = spk_num.group(1) if spk_num else None
        if label and label not in speaker_map:
            for name in names:
                if name in text:
                    speaker_map[label] = name
                    break
        label_name = speaker_map.get(label, f"Speaker {label}" if label else orig_label)
        out_lines.append(f"{timestamp} {label_name}: {text.strip()}")
    return "\n".join(out_lines)

def send_slack_error(record_id: str, error_message: str):
    if not SLACK_WEBHOOK_URL: return
    payload = {"text": f":rotating_light: *Transcription Error Alert*\n• *Record ID:* `{record_id}`\n• *Error:* `{error_message}`"}
    try:
        requests.post(SLACK_WEBHOOK_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=10)
    except Exception as e:
        logging.warning(f"Failed to send Slack error alert: {e}")

def send_slack_success(record_id: str, doc_url: str):
    if not SLACK_WEBHOOK_URL: return
    payload = {"text": f":white_check_mark: *Transcription Complete*\n• *Record ID:* `{record_id}`\n• *Document:* {doc_url}"}
    try:
        requests.post(SLACK_WEBHOOK_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=10)
    except Exception: pass

def update_progress(table, record_id, status, progress_pct=None):
    try:
        table.update(record_id, {STATUS_FIELD: status})
        logging.info(f"Updated status to: {status}")
    except Exception as e:
        logging.warning("Failed to update progress: %s", e)

def slugify(txt: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', txt).strip('_')

def extract_drive_id(url: str) -> Optional[str]:
    m = re.search(r"/file/d/([^/]+)/", url)
    return m.group(1) if m else None

def cache_key(url: str, fid: Optional[str]) -> str:
    if fid: return f"drive_{fid}"
    return "url_" + sha1(url.encode("utf-8")).hexdigest()[:16]

def get_creds() -> Optional["Credentials"]:
    if not GOOGLE_OK or not os.path.exists(CREDENTIALS_JSON): return None
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
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_JSON, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_JSON, "w") as token:
            token.write(creds.to_json())
    return creds

def drive_download(file_id: str, dest: Path, creds: "Credentials"):
    http = httplib2.Http()
    drive = build("drive", "v3", http=AuthorizedHttp(creds, http=http), cache_discovery=False)
    request = drive.files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

def http_download(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

def download_any(url: str, tmp_dest: Path, creds: Optional["Credentials"]) -> Path:
    fid = extract_drive_id(url)
    key = cache_key(url, fid)
    cache_path = CACHE_DIR / key
    if USE_CACHE and cache_path.exists() and not REFRESH_CACHE:
        logging.info("Using cached file: %s", cache_path.name)
        return cache_path
    logging.info("Cache miss or refresh. Downloading...")
    if fid and creds:
        drive_download(fid, tmp_dest, creds)
    else:
        http_download(url, tmp_dest)
    shutil.copy2(tmp_dest, cache_path)
    return cache_path

def prepare_audio(src: Path, name_field: str, record_id: str) -> Path:
    base = f"{slugify(name_field)}_{record_id[:6]}"
    dest = src.with_name(base + ".wav")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dest)]
    logging.info("ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return dest

def run_whisperx(input_audio: Path, table, record_id: str) -> dict:
    logging.info("Loading WhisperX model (%s)", WHISPERX_MODEL)
    update_progress(table, record_id, "Loading Model", 10)
    model = whisperx.load_model(WHISPERX_MODEL, device=DEVICE, language="en")
    
    logging.info("Transcribing...")
    update_progress(table, record_id, "Transcribing", 25)
    result = model.transcribe(str(input_audio))
    
    logging.info("Aligning words...")
    update_progress(table, record_id, "Aligning", 50)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, str(input_audio), DEVICE)

    if DO_DIARIZATION and HF_TOKEN:
        logging.info("Running diarization...")
        update_progress(table, record_id, "Diarizing", 75)
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(str(input_audio))
        result = whisperx.assign_word_speakers(diarize_segments, result)
    
    update_progress(table, record_id, "Formatting", 90)
    return result

def normalize_speakers(result: dict) -> dict:
    segs = result.get("segments", [])
    if not any(s.get("speaker") for s in segs):
        for s in segs: s["speaker"] = "Speaker 1"
        return result
    mapping, next_id = {}, 1
    for s in segs:
        raw = s.get("speaker")
        if raw and raw not in mapping:
            mapping[raw] = f"Speaker {next_id}"
            next_id += 1
        if raw: s["speaker"] = mapping[raw]
    return result

def format_transcript(result: dict, pause_sec: float) -> str:
    segs = result.get("segments", [])
    merged = []
    cur = None
    for s in segs:
        spk = s.get("speaker", "Speaker 1")
        start, end = s.get("start", 0.0), s.get("end", 0.0)
        text = s.get("text", "").strip()
        if cur and cur["speaker"] == spk and (start - cur["end"]) <= pause_sec:
            cur["text"] += " " + text
            cur["end"] = end
        else:
            if cur: merged.append(cur)
            cur = {"speaker": spk, "start": start, "end": end, "text": text}
    if cur: merged.append(cur)
    lines = []
    for m in merged:
        ts = time.strftime('%H:%M:%S', time.gmtime(m['start']))
        lines.append(f"[{ts}] {m['speaker']}: {m['text']}")
    return "\n".join(lines)

def google_upload_doc(title: str, text_content: str, creds: Optional["Credentials"]) -> Optional[str]:
    if not creds: return None
    try:
        http = httplib2.Http()
        authed_http = AuthorizedHttp(creds, http=http)
        docs_service = build("docs", "v1", http=authed_http, cache_discovery=False)
        drive_service = build("drive", "v3", http=authed_http, cache_discovery=False)
        
        doc = docs_service.documents().create(body={"title": title}).execute()
        doc_id = doc["documentId"]
        
        requests_body = [{'insertText': {'location': {'index': 1}, 'text': text_content}}]
        docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests_body}).execute()
        
        drive_service.files().update(fileId=doc_id, addParents=DRIVE_FOLDER_ID, removeParents="root").execute()
        
        return f"https://docs.google.com/document/d/{doc_id}/edit"
    except Exception as e:
        logging.error(f"Google Docs upload failed: {e}")
        return None

# ─── FLASK APP & MAIN LOGIC ─────────────────────────────────
app = Flask(__name__)

def perform_transcription(record_id: str):
    setup_logging(record_id)
    log = logging.getLogger("transcriber")
    log.info(f"Starting transcription for record {record_id}")
    
    table = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
    
    try:
        rec = table.get(record_id)
        update_progress(table, record_id, "Processing", 5)
        
        name_field = rec["fields"].get(NAME_FIELD, f"rec_{record_id[:6]}")
        url = rec["fields"].get(AUDIO_LINK_FIELD)
        log.info(f"Processing: {name_field}")

        if not url: raise ValueError(f"No URL in field '{AUDIO_LINK_FIELD}'")

        creds = get_creds()
        if not creds: log.warning("No Google credentials, Drive/Docs features may be limited.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            temp_download = tmpdir / "downloaded_file"
            
            log.info(f"Downloading {url}")
            src_path = download_any(url, temp_download, creds)
            
            audio_path = prepare_audio(src_path, name_field, record_id)
            result = run_whisperx(audio_path, table, record_id)
            
            result = normalize_speakers(result)
            raw = format_transcript(result, pause_sec=PAUSE_SEC)
            names = extract_names(raw)
            transcript_text = assign_names(raw, names)
            
            log.info("Uploading transcript...")
            doc_url = google_upload_doc(name_field, transcript_text, creds)
            
            if not doc_url: raise RuntimeError("Failed to upload transcript to Google Docs/Drive.")
            
            table.update(record_id, {
                STATUS_FIELD: "Transcribed",
                TRANSCRIPT_URL_FLD: doc_url
            })
            log.info("Successfully updated Airtable record")
            send_slack_success(record_id, doc_url)

    except Exception as e:
        error_msg = str(e)
        log.error(f"Transcription failed for {record_id}: {error_msg}", exc_info=True)
        send_slack_error(record_id, error_msg)
        try:
            table.update(record_id, {STATUS_FIELD: "Error"})
        except Exception as update_err:
            log.error(f"Failed to update status to Error: {update_err}")
        raise e

@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    data = request.get_json()
    if not data or "record_id" not in data:
        return jsonify({"error": "Missing 'record_id' in request body"}), 400

    record_id = data.get("record_id")
    if not record_id:
        return jsonify({"error": "record_id cannot be null or empty"}), 400

    logging.info(f"Received transcription request for record: {record_id}")

    try:
        perform_transcription(record_id)
        return jsonify({"status": "success", "record_id": record_id}), 200
    except Exception as e:
        logging.error(f"Error during transcription for {record_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))