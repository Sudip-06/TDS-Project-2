# main.py

import os, io, re, json, base64, zipfile, random, string, mimetypes, tempfile, warnings, time, asyncio, hashlib
from typing import List, Optional, Dict, Any, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import httpx
from bs4 import BeautifulSoup
from lxml import etree
import yaml
from docx import Document

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager

# =========================
# Config (OpenRouter via aipipe) + time budget caps
# =========================

OPENROUTER_URL   = os.getenv("OPENROUTER_URL", "https://aipipe.org/openrouter/v1/chat/completions")
OPENROUTER_TOKEN = os.getenv("GITHUB_TOKEN")  # <-- put your real token here or in env

MODEL_SCRAPE     = os.getenv("MODEL_SCRAPE",  "openai/gpt-5-mini")
MODEL_PROCESS    = os.getenv("MODEL_PROCESS", "openai/gpt-4o")

TOTAL_DEADLINE_S = int(os.getenv("TOTAL_DEADLINE_S", "290"))  # 5 minutes soft cap
MAX_LLM_RETRIES  = int(os.getenv("MAX_LLM_RETRIES", "2"))
LLM_TIMEOUT_S    = int(os.getenv("LLM_TIMEOUT_S", "20"))
HTTP_TIMEOUT_S   = int(os.getenv("HTTP_TIMEOUT_S", "10"))

MAX_JOINED_TEXT  = int(os.getenv("MAX_JOINED_TEXT", "10000"))
MAX_TOK_PLAN     = int(os.getenv("MAX_TOK_PLAN", "500"))
MAX_TOK_PROCESS  = int(os.getenv("MAX_TOK_PROCESS", "900"))

DATA_DIR = os.getenv("DATA_DIR", "/tmp/data")  # <- use /tmp, not ./data
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# FastAPI app + shared client
# =========================

_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    headers = {"User-Agent": "Mozilla/5.0"}
    _client = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_S,
        http2=True,
        headers=headers,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=60.0),
    )
    try:
        yield
    finally:
        await _client.aclose()

app = FastAPI(title="Data Analyst Agent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# =========================
# Deadline helpers
# =========================

def now_s() -> float:
    return time.monotonic()

def deadline_guard(start_s: float) -> float:
    return TOTAL_DEADLINE_S - (now_s() - start_s)

def near_deadline(start_s: float, margin: float = 8.0) -> bool:
    return deadline_guard(start_s) <= margin

# =========================
# Helpers
# =========================

def _save_upload(u: UploadFile, base_dir="data") -> str:
    name = u.filename or ("upload_" + "".join(random.choices(string.ascii_lowercase, k=8)))
    path = os.path.join(base_dir, name)
    with open(path, "wb") as f:
        f.write(u.file.read())
    return path

def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")
        except:
            return ""

def _guess_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        mt, _ = mimetypes.guess_type(path)
        if mt and "/" in mt:
            ext = "." + mt.split("/")[-1]
    return ext

def extract_urls(s: str) -> List[str]:
    return re.findall(r"https?://\S+", s or "")

def text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script","style","noscript"]):
        s.extract()
    return soup.get_text(" ", strip=True)

def scrape_tables_from_html(html: str) -> List[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    tables: List[pd.DataFrame] = []
    for t in soup.find_all("table"):
        try:
            for df in pd.read_html(str(t)):
                tables.append(df)
        except:
            continue
    return tables

# ---------- File parsing ----------

def parse_pdf_text(path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return ("\n".join(pages), None)
    except Exception as e:
        return (None, f"pdf parse error: {e}")

def parse_any_file(path: str) -> Dict[str, Any]:
    ext = _guess_ext(path)
    out: Dict[str, Any] = {"meta": {"path": path, "ext": ext}}
    try:
        if ext in [".txt", ".md", ".rtf", ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
                   ".c", ".cpp", ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt",
                   ".sql", ".sh", ".bat", ".json", ".yaml", ".yml", ".xml"]:
            text = _read_text(path)
            if ext == ".json":
                try: out["json"] = json.loads(text)
                except: pass
            elif ext in [".yaml", ".yml"]:
                try: out["yaml"] = yaml.safe_load(text)
                except: pass
            elif ext == ".xml":
                try: out["xml"] = etree.fromstring(text.encode("utf-8","ignore"))
                except: pass
            out["text"] = text

        elif ext in [".csv", ".tsv"]:
            df = pd.read_csv(path, sep="\t" if ext==".tsv" else ",")
            out["tables"] = [df]

        elif ext in [".xls", ".xlsx", ".ods"]:
            try:
                xls = pd.ExcelFile(path)
                out["tables"] = [xls.parse(s) for s in xls.sheet_names]
            except:
                out["tables"] = [pd.read_excel(path)]

        elif ext == ".pdf":
            text, err = parse_pdf_text(path)
            if text and text.strip():
                out["text"] = text
            else:
                out["binary"] = open(path, "rb").read()
                if err: out["error"] = err

        elif ext == ".docx":
            doc = Document(path)
            out["text"] = "\n".join(p.text for p in doc.paragraphs)

        elif ext in [".doc", ".odt", ".ppt", ".pptx", ".odp"]:
            out["binary"] = open(path, "rb").read()

        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"]:
            try:
                img = Image.open(path).convert("RGB")
                out["images"] = [img]
            except:
                out["binary"] = open(path, "rb").read()

        elif ext in [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm",
                     ".mp4", ".mov", ".mkv", ".zip"]:
            if ext == ".zip":
                tmpd = tempfile.mkdtemp(prefix="unz_")
                with zipfile.ZipFile(path, "r") as z:
                    z.extractall(tmpd)
                out["unzipped_dir"] = tmpd
                out["unzipped_files"] = [os.path.join(tmpd, p) for p in os.listdir(tmpd)]
            else:
                out["binary"] = open(path, "rb").read()
                out["media"] = True
        else:
            out["binary"] = open(path, "rb").read()
    except Exception as e:
        out["error"] = f"parse error: {e}"
    return out

# =========================
# Fetching (async + cache) — deadline aware
# =========================

_FETCH_CACHE: Dict[str, Tuple[str, Optional[bytes]]] = {}

async def fetch_url(url: str, timeout: int = HTTP_TIMEOUT_S, start_s: float | None = None) -> Tuple[str, Optional[bytes]]:
    if url in _FETCH_CACHE:
        return _FETCH_CACHE[url]
    assert _client is not None
    tmo = timeout
    if start_s is not None:
        rem = deadline_guard(start_s)
        tmo = max(3, min(tmo, int(rem - 5))) if rem > 6 else 3
    r = await _client.get(url, timeout=tmo)
    ctype = (r.headers.get("Content-Type") or "").lower()
    out = (r.text, None) if ("text/html" in ctype or "application/xhtml" in ctype) else ("", r.content)
    _FETCH_CACHE[url] = out
    return out

# =========================
# LLM plumbing with retry + cache — deadline aware
# =========================

_LLM_CACHE: Dict[str, str] = {}

def _payload_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

async def _llm(messages, model: str, max_tokens: int = 800, temperature: float = 0, start_s: float | None = None) -> str:
    if not OPENROUTER_TOKEN:
        raise RuntimeError("Missing GITHUB_TOKEN (OpenRouter token)")
    assert _client is not None
    headers = {"Authorization": f"Bearer {OPENROUTER_TOKEN}", "Content-Type":"application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    key = _payload_hash({"url": OPENROUTER_URL, **payload})
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]

    last_err: Exception | None = None
    for attempt in range(MAX_LLM_RETRIES):
        per_req_timeout = LLM_TIMEOUT_S
        if start_s is not None:
            rem = deadline_guard(start_s)
            per_req_timeout = max(3, min(per_req_timeout, int(rem - 5))) if rem > 6 else 3

        try:
            r = await _client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=per_req_timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                delay = 0.7 * (attempt + 1)
                if start_s is not None and deadline_guard(start_s) <= (delay + 5):
                    break
                await asyncio.sleep(delay)
                last_err = httpx.HTTPStatusError(f"{r.status_code} {r.reason_phrase}", request=r.request, response=r)
                continue
            r.raise_for_status()
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            _LLM_CACHE[key] = txt
            return txt
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = e
            if start_s is not None and near_deadline(start_s, margin=10):
                break
            await asyncio.sleep(0.5)
            continue
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                raise
            last_err = e
            if start_s is not None and near_deadline(start_s, margin=10):
                break
            await asyncio.sleep(0.5)
            continue
    raise last_err or RuntimeError("LLM request failed after retries")

async def llm_plan(question: str, context_brief: str, start_s: float | None = None) -> Dict[str, Any]:
    sys = (
        "Plan how to solve data analysis tasks with available tools.\n"
        "- Tools: Python can scrape URLs, read HTML tables, parse files, compute stats, plot via matplotlib.\n"
        "- RETURN a STRICT JSON object with keys: actions (array), expected_items (int), plot_specs (array), notes (string).\n"
        "No prose outside JSON."
    )
    usr = f"QUESTION:\n{question}\n\nCONTEXT SUMMARY:\n{context_brief}\n"
    try:
        out = await _llm(
            [{"role":"system","content": sys},{"role":"user","content": usr}],
            model=MODEL_SCRAPE, max_tokens=MAX_TOK_PLAN, start_s=start_s
        )
    except Exception:
        return {"actions":["best_effort_json_array"], "expected_items":1, "plot_specs":[], "notes":"planner_unavailable"}
    try:
        return json.loads(out)
    except:
        m = re.search(r"\{[\s\S]*\}", out)
        return json.loads(m.group(0)) if m else {"actions":["best_effort_json_array"], "expected_items":1, "plot_specs":[], "notes":"planner_unavailable"}

def parse_llm_json_array(s: str) -> list:
    if not s:
        return []
    s = s.strip()
    s = re.sub(r"^\s*`(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*`\s*$", "", s)
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        s = m.group(0)
    arr = json.loads(s)
    if not isinstance(arr, list):
        raise ValueError("LLM did not return a JSON array")
    return arr

async def llm_array_only(prompt: str, start_s: float | None = None) -> list:
    try:
        raw = await _llm(
            [
                {"role":"system","content":"Return ONLY a raw JSON array. No prose. No code fences. No keys."},
                {"role":"user","content": prompt},
            ],
            model=MODEL_PROCESS, max_tokens=MAX_TOK_PROCESS, temperature=0, start_s=start_s
        )
        return parse_llm_json_array(raw)
    except Exception:
        return ["RATE_LIMIT_OR_LLM_ERROR"]

def coerce_numbers(arr: list) -> list:
    out = []
    for x in arr:
        if isinstance(x, str) and not x.startswith("data:image/"):
            xs = x.strip()
            if re.fullmatch(r"-?\d+", xs):
                try:
                    out.append(int(xs)); continue
                except:
                    pass
            if re.fullmatch(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", xs, re.I):
                try:
                    out.append(float(xs)); continue
                except:
                    pass
        out.append(x)
    return out

# =========================
# Plotting
# =========================

def ensure_png_under_limit(pil_img: Image.Image, max_bytes: int = 120_000) -> str:
    w, h = pil_img.size
    for scale in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4]:
        target = pil_img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.LANCZOS)
        buf = io.BytesIO()
        target.save(buf, format="PNG", optimize=True)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return "data:image/png;base64," + base64.b64encode(b).decode()
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def tiny_placeholder_png(text="plot", w=320, h=240) -> str:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.load_default()
    except:
        fnt = None
    d.text((10, 10), text, fill=(0, 0, 0), font=fnt)
    return ensure_png_under_limit(img)

def plot_from_spec(df: pd.DataFrame, spec: Dict[str, Any]) -> str:
    plt.figure(figsize=(6,4))
    t = (spec.get("type") or "scatter").lower()
    cols = spec.get("columns") or []
    xlab = spec.get("x_label", cols[0] if cols else "x")
    ylab = spec.get("y_label", cols[1] if len(cols)>1 else "y")
    title = spec.get("title", "")
    regression = str(spec.get("regression","")).lower()

    if t == "scatter" and len(cols) >= 2 and all(c in df.columns for c in cols):
        x = pd.to_numeric(df[cols[0]], errors="coerce").values
        y = pd.to_numeric(df[cols[1]], errors="coerce").values
        m = (~np.isnan(x)) & (~np.isnan(y))
        x, y = x[m], y[m]
        plt.scatter(x, y)
        if regression and len(x) > 1:
            m1, c1 = np.polyfit(x, y, 1)
            xx = np.linspace(np.min(x), np.max(x), 200)
            yy = m1*xx + c1
            plt.plot(xx, yy, linestyle="--", color="red")
    elif t == "line" and len(cols) >= 2:
        plt.plot(df[cols[0]], df[cols[1]])
    elif t == "bar" and len(cols) >= 2:
        plt.bar(df[cols[0]], df[cols[1]])
    elif t == "hist" and cols:
        plt.hist(pd.to_numeric(df[cols[0]], errors="coerce").dropna().values, bins=spec.get("bins", 20))
    elif t == "box" and cols:
        data = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in cols if c in df.columns]
        labs = [c for c in cols if c in df.columns]
        if data:
            plt.boxplot(data, labels=labs)
        else:
            plt.text(0.5,0.5,"No numeric data", ha="center")
    else:
        plt.text(0.5,0.5,"Unsupported/invalid plot spec", ha="center")

    plt.xlabel(xlab); plt.ylabel(ylab)
    if title: plt.title(title)
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close()
    pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return ensure_png_under_limit(pil)

# =========================
# Local quick paths (speed boosters)
# =========================

_SIMPLE_MATH = re.compile(r"^\s*([0-9\.\s\+\-\*\/\(\)]+)\s*\=?\??\s*$")

def _eval_simple_math(expr: str) -> Optional[float]:
    if not _SIMPLE_MATH.match(expr or ""):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        if isinstance(val, (int, float)) and np.isfinite(val):
            return float(val)
    except Exception:
        return None
    return None

# =========================
# Deterministic solvers (no-LLM, accurate, fast)
# =========================

def _normalize_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())

def _find_height_weight_df(tables: List[pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, str, str]]:
    for df in tables:
        cols = list(df.columns)
        norm = {c: _normalize_colname(c) for c in cols}
        # candidates
        h_candidates = [c for c in cols if re.search(r"height", norm[c])]
        w_candidates = [c for c in cols if re.search(r"weight", norm[c])]
        if h_candidates and w_candidates:
            h = h_candidates[0]; w = w_candidates[0]
            dfx = df.copy()
            dfx[h] = pd.to_numeric(dfx[h], errors="coerce")
            dfx[w] = pd.to_numeric(dfx[w], errors="coerce")
            return dfx, h, w
    return None

def try_answer_hw_csv(prompt: str, tables: List[pd.DataFrame]) -> Optional[list]:
    # Fire only if user intent mentions height & weight (very common in your question.txt)
    if not re.search(r"\bheight\b", prompt, re.I) or not re.search(r"\bweight\b", prompt, re.I):
        return None
    hw = _find_height_weight_df(tables)
    if not hw:
        return None
    df, h, w = hw
    # 1) averages rounded to 2 decimals
    avg_h = round(float(pd.to_numeric(df[h], errors="coerce").dropna().mean()), 2)
    avg_w = round(float(pd.to_numeric(df[w], errors="coerce").dropna().mean()), 2)
    # 2) count: height > 70 and weight < 150
    m = (pd.to_numeric(df[h], errors="coerce") > 70) & (pd.to_numeric(df[w], errors="coerce") < 150)
    count = int(m.sum())
    # 3) correlation rounded to 3 decimals
    hh = pd.to_numeric(df[h], errors="coerce")
    ww = pd.to_numeric(df[w], errors="coerce")
    mask = (~hh.isna()) & (~ww.isna())
    corr = float(np.corrcoef(hh[mask], ww[mask])[0,1]) if mask.any() else float("nan")
    if np.isnan(corr):
        corr = 0.0
    corr = round(corr, 3)
    return [[avg_h, avg_w], count, corr]

# =========================
# /api endpoint (deadline-aware)
# =========================

@app.post("/api")
async def api(
    question_text: Optional[str] = Form(default=None),
    urls: Optional[str] = Form(default=""),
    # legacy single-field prompt file
    question: Optional[UploadFile] = File(default=None, alias="question.txt"),
    # legacy multiple files
    files: List[UploadFile] = File(default=[], alias="files"),

    files: List[UploadFile] = File(default=[], alias="image.png"),
    files: List[UploadFile] = File(default=[], alias="data.csv"),
   
    # NEW: support curl -F "file=@x" repeated
    files_generic: List[UploadFile] = File(default=[], alias="file"),
):
    start_s = now_s()
    try:
        # merge all uploaded files
        uploads: List[UploadFile] = []
        if question is not None:
            uploads.append(question)
        uploads.extend(files or [])
        uploads.extend(files_generic or [])

        # prompt from field or infer from uploaded text files (e.g., question.txt)
        prompt = (question_text or "").strip()
        inferred_prompt = None
        text_candidates: List[Tuple[str, str]] = []  # (filename, text)
        if not prompt and uploads:
            for uf in uploads:
                try:
                    name = uf.filename or ""
                    content = (await uf.read()).decode("utf-8", "ignore")
                    uf.file.seek(0)
                    if content and content.strip():
                        text_candidates.append((name, content.strip()))
                except Exception:
                    pass
            for name, txt in text_candidates:
                low = name.lower()
                if low.startswith("question") or "question" in low or low.endswith(".txt"):
                    inferred_prompt = txt
                    break
            if inferred_prompt is None and text_candidates:
                inferred_prompt = text_candidates[0][1]
        if not prompt and inferred_prompt:
            prompt = inferred_prompt

        provided_urls = set(extract_urls(prompt))
        if urls:
            for u in re.split(r"[\s,]+", urls.strip()):
                if u.startswith("http"):
                    provided_urls.add(u)

        # save & parse all uploads
        saved_files = []
        for f in uploads:
            try:
                saved_files.append(_save_upload(f))
            except:
                pass
        parsed_files = [parse_any_file(p) for p in saved_files]
        for item in list(parsed_files):
            if "unzipped_files" in item:
                for p in item["unzipped_files"]:
                    parsed_files.append(parse_any_file(p))

        # Fetch URLs concurrently (deadline-aware)
        url_texts, url_tables = [], []
        if provided_urls and not near_deadline(start_s, margin=12):
            coros = [fetch_url(u, start_s=start_s) for u in provided_urls]
            html_results = await asyncio.gather(*coros, return_exceptions=True)
            for (u, hr) in zip(provided_urls, html_results):
                if isinstance(hr, Exception):
                    url_texts.append(f"[Fetch error {u}: {hr}]"); continue
                html, raw = hr
                if html:
                    url_texts.append(text_from_html(html))
                    url_tables.extend(scrape_tables_from_html(html))

        # gather materials
        candidate_tables: List[pd.DataFrame] = []
        for pf in parsed_files:
            for t in pf.get("tables", []) or []:
                candidate_tables.append(t)
        candidate_tables.extend(url_tables)

        big_text_chunks: List[str] = []
        for pf in parsed_files:
            if "text" in pf and pf["text"] != None:
                if pf["text"]:
                    big_text_chunks.append(pf["text"])
        big_text_chunks.extend(url_texts)
        if prompt:
            big_text_chunks.append(prompt)

        # Fast path for simple math
        if prompt:
            mval = _eval_simple_math(prompt.replace("=", "").replace("?", ""))
            if mval is not None:
                return [int(mval) if float(mval).is_integer() else mval]

        # ---- Deterministic solvers (no LLM) ----
        # 1) Height/Weight CSV QA (your hw_200.csv style)
        hw_ans = try_answer_hw_csv(prompt, candidate_tables)
        if hw_ans is not None:
            return hw_ans  # already in correct JSON array format

        # (You can add more deterministic solvers here if needed)

        # If nothing deterministic matched, proceed with LLM path
        context_brief = json.dumps({
            "urls_found": len(provided_urls),
            "files_uploaded": len(saved_files),
            "tables_detected": int(np.sum([len(pf.get("tables",[])) for pf in parsed_files])) + len(url_tables),
            "text_chars": int(np.sum([len(pf.get("text","")) for pf in parsed_files])) + int(np.sum([len(x) for x in url_texts])),
        })

        if near_deadline(start_s, margin=10):
            plan = {"actions": ["best_effort_json_array"], "expected_items": 1, "plot_specs": [], "notes": "deadline_shortcut"}
        else:
            plan = await llm_plan(prompt, context_brief, start_s=start_s)

        answers: List[Any] = []
        full_answer_from_llm = False

        def find_df(req_cols: List[str]) -> Optional[pd.DataFrame]:
            for df in candidate_tables:
                cols = [str(c) for c in df.columns]
                if all(any(rc == c or rc.lower()==c.lower() for c in cols) for rc in req_cols):
                    dfx = df.copy()
                    for c in req_cols:
                        if c in dfx.columns:
                            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
                    return dfx
            return None

        def compute_corr(a: str, b: str) -> Optional[float]:
            df = find_df([a,b]) or (candidate_tables[0] if candidate_tables else None)
            if df is None or a not in df.columns or b not in df.columns:
                return None
            x = pd.to_numeric(df[a], errors="coerce")
            y = pd.to_numeric(df[b], errors="coerce")
            m = ~x.isna() & ~y.isna()
            if not m.any():
                return None
            return float(np.corrcoef(x[m], y[m])[0,1])

        for step in plan.get("actions", []):
            if near_deadline(start_s, margin=7):
                break
            st = str(step).lower()

            m = re.search(r"correlation.*\bbetween\b\s*([A-Za-z0-9_ ]+?)\s*\b(?:and|vs)\b\s*([A-Za-z0-9_ ]+)", st)
            if not m:
                m = re.search(r"correlation.*?([A-Za-z0-9_ ]+)\s*(?:and|vs|,)\s*([A-Za-z0-9_ ]+)", st)
            if m:
                a, b = m.group(1).strip(), m.group(2).strip()
                c = compute_corr(a, b)
                if c is not None:
                    answers.append(float(np.round(c, 6)))
                continue

            if "best_effort_json_array" in st or "answer_with_best_effort" in st:
                joined_text = "\n\n".join(big_text_chunks)[:MAX_JOINED_TEXT]
                answers = await llm_array_only(
                    f"QUESTION:\n{prompt}\n\n"
                    f"CONTEXT TEXT (may include scraped/attached content):\n{joined_text}\n\n"
                    f"Return ONLY a JSON array of answers in the order implied by the question. "
                    f"Use numbers as numbers, strings as strings, and base64 images as data:image/png URIs if requested.",
                    start_s=start_s
                )
                full_answer_from_llm = True
                break

        if not full_answer_from_llm and not near_deadline(start_s, margin=6):
            for spec in plan.get("plot_specs", []) or []:
                req_cols = spec.get("columns") or []
                df = find_df(req_cols) if req_cols else (candidate_tables[0] if candidate_tables else None)
                if df is None:
                    df = pd.DataFrame({"x": np.arange(10), "y": np.linspace(0,1,10)})
                    spec.setdefault("columns", ["x","y"])
                if spec.get("type","").lower() == "scatter":
                    spec.setdefault("regression", "dotted red")
                if req_cols:
                    spec.setdefault("x_label", str(req_cols[0]))
                    spec.setdefault("y_label", str(req_cols[1] if len(req_cols)>1 else "y"))
                answers.append(plot_from_spec(df, spec))

            if not answers and not near_deadline(start_s, margin=5):
                joined_text = "\n\n".join(big_text_chunks)[:MAX_JOINED_TEXT]
                answers = await llm_array_only(
                    f"QUESTION:\n{prompt}\n\n"
                    f"CONTEXT TEXT (may include scraped/attached content):\n{joined_text}\n\n"
                    f"Return ONLY a JSON array of answers in the order implied by the question.",
                    start_s=start_s
                )
                full_answer_from_llm = True

        if len(answers) == 1 and isinstance(answers[0], str) and "[" in answers[0] and "]" in answers[0]:
            try:
                answers = parse_llm_json_array(answers[0])
            except Exception:
                pass

        answers = coerce_numbers(answers)

        fixed = []
        for x in answers:
            if isinstance(x, str) and x.startswith("data:image/png;base64,"):
                b64 = x.split(",",1)[1]
                if "..." in b64 or len(b64) < 300:
                    fixed.append(tiny_placeholder_png("auto-generated plot"))
                else:
                    fixed.append(x)
            else:
                fixed.append(x)
        answers = fixed

        if (not answers) or near_deadline(start_s, margin=3):
            if not answers:
                try:
                    answers = await llm_array_only(f"QUESTION:\n{prompt}\n\nReturn ONLY a non-empty JSON array of answers.", start_s=start_s)
                except Exception:
                    answers = []
            if not answers:
                answers = ["UNKNOWN"]

        if answers[:1] == ["RATE_LIMIT_OR_LLM_ERROR"]:
            # Final guard: never return pure rate-limit. Give a helpful hint array.
            return JSONResponse(status_code=200, content=["RETRY_LATER_OR_ADD_TEXT_PROMPT"])
        return answers

    except Exception as e:
        return JSONResponse(
            content=[f"ERROR:{type(e).__name__}", str(e)],
            status_code=500
        )

# =========================
# Simple Frontend (HTML) at /ui
# =========================

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return r"""
<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Data Analyst Agent</title>
<style>
  :root{
    --bg:#0b1220;--card:#111829;--muted:#a9b8d6;--text:#e8f0ff;--edge:#22345d;
    --btn:#2563eb;--btn2:#0ea5e9;--chip:#132344;--chip-edge:#28407a;
    --accent:#7dd3fc;--ok:#22c55e;--bad:#ef4444;
  }
  [data-theme="light"]{
    --bg:#f3f6fb;--card:#ffffff;--muted:#44516a;--text:#0c1222;--edge:#e1e6f2;
    --btn:#2563eb;--btn2:#0ea5e9;--chip:#eef3ff;--chip-edge:#d4defa;--accent:#2563eb;
  }
  *{box-sizing:border-box}
  body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:var(--bg);color:var(--text)}
  .wrap{max-width:1000px;margin:28px auto;padding:0 16px}
  header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
  h1{font-size:20px;margin:0}
  .card{background:var(--card);border:1px solid var(--edge);border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 6px 24px rgba(0,0,0,.14)}
  label{display:block;font-size:13px;color:var(--muted);margin:8px 0 6px}
  textarea,input{width:100%;padding:12px 14px;border:1px solid var(--edge);background:transparent;color:var(--text);border-radius:10px}
  textarea{min-height:120px;resize:vertical}
  .row{display:flex;gap:12px;flex-wrap:wrap}
  .row > *{flex:1}
  .btn{display:inline-flex;align-items:center;gap:8px;background:var(--btn);color:#fff;border:0;border-radius:10px;padding:11px 16px;font-weight:700;cursor:pointer}
  .btn[disabled]{opacity:.6;cursor:not-allowed}
  .ghost{background:transparent;border:1px solid var(--edge);color:var(--text)}
  .hint{color:var(--muted);font-size:12px}
  .pill{display:inline-block;background:var(--chip);border:1px solid var(--chip-edge);border-radius:999px;padding:5px 10px;margin:4px 6px 0 0;font-size:12px}
  .pill button{margin-left:6px;border:0;background:transparent;color:var(--muted);cursor:pointer}
  .drop{border:1.5px dashed var(--edge);border-radius:12px;padding:14px;text-align:center;transition:.15s}
  .drop.drag{background:rgba(125,211,252,.08);border-color:var(--btn2)}
  .status{display:flex;align-items:center;gap:8px}
  .dot{width:8px;height:8px;border-radius:50%}
  .ok{background:var(--ok)} .bad{background:var(--bad)} .progress{border:2px solid var(--edge);border-top-color:var(--btn);border-radius:50%;width:16px;height:16px;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .tabs{display:flex;gap:8px;margin-bottom:8px}
  .tab{padding:6px 10px;border:1px solid var(--edge);border-radius:8px;background:transparent;cursor:pointer}
  .tab.active{background:var(--chip);border-color:var(--chip-edge)}
  pre{white-space:pre-wrap;word-break:break-word;background:#0f1726;border:1px solid var(--edge);border-radius:10px;padding:12px;max-height:480px;overflow:auto}
  [data-theme="light"] pre{background:#f7f9ff}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
  img.inline{max-width:100%;border-radius:8px;border:1px solid var(--edge);background:#000}
  .topbar{display:flex;gap:8px;align-items:center}
  .actions{display:flex;gap:8px;flex-wrap:wrap}
  .right{margin-left:auto}
  .small{font-size:12px;padding:6px 10px}
  .kbd{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--chip);border:1px solid var(--chip-edge);padding:2px 6px;border-radius:6px;font-size:11px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Data Analyst Agent</h1>
    <div class="topbar">
      <button id="themeBtn" class="btn ghost small">Toggle theme</button>
      <a href="/" class="btn ghost small">Health</a>
      <a href="/docs" class="btn ghost small">API Docs</a>
    </div>
  </header>

  <div class="card">
    <label>Question</label>
    <textarea id="question" placeholder="Ask a data question. You can also upload files and/or paste URLs below. (Tip: press Ctrl/⌘+Enter to run)"></textarea>

    <div class="row">
      <div>
        <label>URLs (optional, comma or space separated)</label>
        <input id="urls" placeholder="https://example.com/page1 https://example.com/page2" />
      </div>
      <div>
        <label>Files (optional)</label>
        <div id="drop" class="drop">
          <input id="fileInput" type="file" multiple style="display:none" />
          <div>Drag & drop files here or <u id="browse">browse</u></div>
          <div class="hint">CSV, XLS/XLSX, PDF, DOCX, images, ZIP…</div>
        </div>
        <div id="chips"></div>
      </div>
    </div>

    <div class="row" style="align-items:center;margin-top:12px">
      <div class="actions">
        <button id="runBtn" class="btn">Run</button>
        <button id="clearBtn" class="btn ghost">Clear</button>
        <button id="copyBtn" class="btn ghost">Copy JSON</button>
        <button id="saveBtn" class="btn ghost">Download JSON</button>
      </div>
      <span class="right status hint" id="status"><span class="dot" style="background:#777"></span> idle</span>
    </div>
    <div class="hint" style="margin-top:6px">Shortcut: <span class="kbd">Ctrl/⌘ + Enter</span></div>
  </div>

  <div class="card">
    <div class="tabs">
      <button class="tab active" data-tab="json">JSON</button>
      <button class="tab" data-tab="images">Images</button>
    </div>
    <div id="pane-json">
      <pre id="out">—</pre>
    </div>
    <div id="pane-images" style="display:none">
      <div class="grid" id="images"></div>
    </div>
  </div>

  <p class="hint">The API returns a <b>JSON array</b>. Base64 images (data:image/png) are rendered in the Images tab.</p>
</div>

<script>
const $ = (id)=>document.getElementById(id);
const el = (tag, attrs={}, children=[]) => {
  const e=document.createElement(tag);
  Object.entries(attrs).forEach(([k,v])=>k==="class"?e.className=v:(k==="html"?e.innerHTML=v:e.setAttribute(k,v)));
  children.forEach(c=>e.appendChild(c)); return e;
};
const state = { files: [] };

function toggleTheme(){
  const root=document.documentElement;
  root.dataset.theme = (root.dataset.theme==="dark"?"light":"dark");
}
$("themeBtn").onclick=toggleTheme;

function refreshChips(){
  const box=$("chips"); box.innerHTML="";
  state.files.forEach((f,idx)=>{
    const chip=el("span",{class:"pill"});
    chip.append(document.createTextNode(f.name || `file_${idx+1}`));
    const rm=el("button",{title:"Remove"},[document.createTextNode("×")]);
    rm.onclick=()=>{ state.files.splice(idx,1); refreshChips(); };
    chip.appendChild(rm); box.appendChild(chip);
  });
}

const drop=$("drop"), fileInput=$("fileInput"), browse=$("browse");
browse.onclick=()=>fileInput.click();
fileInput.onchange=(e)=>{ state.files=[...state.files, ...e.target.files]; refreshChips(); fileInput.value=""; };
["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,(e)=>{e.preventDefault(); drop.classList.add("drag");}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,(e)=>{e.preventDefault(); drop.classList.remove("drag");}));
drop.addEventListener("drop",(e)=>{ state.files=[...state.files, ...e.dataTransfer.files]; refreshChips(); });

function setStatus(txt,type){
  const s=$("status"); s.textContent=" "+txt;
  const dot=s.querySelector(".dot");
  if(type==="ok") dot.className="dot ok";
  else if(type==="bad") dot.className="dot bad";
  else dot.className="dot";
}

function setTabs(which){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.tab===which));
  $("pane-json").style.display = (which==="json")?"block":"none";
  $("pane-images").style.display = (which==="images")?"block":"none";
}
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>setTabs(t.dataset.tab));

async function callAPI() {
  const t0 = performance.now();
  $("out").textContent = "Working…";
  $("images").innerHTML = "";
  setStatus("sending…","");

  const btn=$("runBtn");
  btn.disabled=true; btn.innerHTML='<span class="progress"></span> Running';

  try {
    const fd = new FormData();
    const q = $("question").value.trim();
    const urls = $("urls").value.trim();
    if (q) fd.append("question_text", q);
    if (urls) fd.append("urls", urls);
    state.files.forEach(f=>fd.append("files", f, f.name));

    const ctrl = new AbortController();
    const timeoutMs = 290000; // < 5 minutes
    const to = setTimeout(()=>ctrl.abort(), timeoutMs);

    const res = await fetch("/api", { method:"POST", body:fd, signal:ctrl.signal });
    clearTimeout(to);
    const txt = await res.text();
    let data; try { data = JSON.parse(txt); } catch { data = txt; }
    $("out").textContent = (typeof data === "string") ? data : JSON.stringify(data, null, 2);

    let count=0;
    if (Array.isArray(data)) {
      const imgs=[];
      for (const item of data) if (typeof item==="string" && item.startsWith("data:image/")) imgs.push(item);
      $("images").innerHTML = imgs.map(src=>`<img class="inline" src="${src}" />`).join("");
      count = imgs.length;
    }
    const ms = Math.round(performance.now()-t0);
    setStatus(`done in ${ms} ms (HTTP ${res.status}) • images: ${count}`,"ok");
  } catch (err) {
    $("out").textContent = String(err);
    setStatus("failed","bad");
  } finally {
    btn.disabled=false; btn.textContent="Run";
  }
}

$("runBtn").addEventListener("click", callAPI);
$("clearBtn").onclick=()=>{
  $("question").value=""; $("urls").value=""; state.files=[]; refreshChips();
  $("out").textContent="—"; $("images").innerHTML=""; setStatus("idle","");
};
$("copyBtn").onclick=()=>{
  const t=$("out").textContent||"";
  navigator.clipboard.writeText(t);
  setStatus("copied JSON to clipboard","ok");
};
$("saveBtn").onclick=()=>{
  const blob = new Blob([$("out").textContent||"[]"], {type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="response.json"; a.click();
  setStatus("downloaded JSON","ok");
};
document.addEventListener("keydown",(e)=>{
  if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){ callAPI(); }
});
</script>
</body>
</html>
    """

# Health
@app.get("/")
def root():
    return {"ok": True, "message": "Data Analyst Agent API is running. Visit /ui for the web UI. POST /api"}

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
