"""
FM LMS Production Service — hosted edition (Gemini free tier)
FastAPI + Supabase (pgvector) + Google Gemini API
Zero-loss ingestion · Ask FM RAG · Learning Paths
"""
import os, io, re, json, time, uuid, hashlib, asyncio, shutil, ssl as sslmod
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

DATABASE_URL  = os.getenv("DATABASE_URL", "")          # Supabase Postgres (pgvector)
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "")
CHAT_MODEL    = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMS    = int(os.getenv("EMBED_DIMS", "768"))
GEMINI_BASE   = "https://generativelanguage.googleapis.com/v1beta"
MAX_MB        = int(os.getenv("MAX_UPLOAD_MB", "200"))
CHUNK_TOK     = int(os.getenv("SECTION_MAX_TOKENS", "2000"))
OVERLAP       = 0.20

CATEGORIES = {
    "HVAC": ["Chillers & Cooling", "Air Handling", "Emergency Response"],
    "Electrical": ["UPS & Generators", "Distribution", "Testing Regimes"],
    "Plumbing": ["Water Quality", "Drainage", "Pumps"],
    "Fire & Life Safety": ["Evacuation", "Devices & Detection", "Compliance"],
    "Emergency Management": ["Incident Command", "Business Continuity", "Drills"],
    "Preventive Maintenance": ["Program Design", "Task Library", "CMMS"],
    "Contractor Management": ["Onboarding", "SLAs & Remedies", "Escalation"],
    "Compliance": ["Statutory Inspections", "Records & Audit", "Standards"],
    "Budget & Finance": ["Operating Budget", "Lifecycle Reserve", "Variance"],
    "FM Fundamentals": ["Role of the FM", "Hard vs Soft Services", "Competency Map"],
    "Team & Leadership": ["Leading FM Teams", "Conflict Management", "Stakeholders"],
    "Operations": ["Opening/Closing", "Shift Handover", "Space Mgmt"],
}
SAFETY_CATS = {"Fire & Life Safety", "Emergency Management"}
STAGE_ORDER = ["FM Fundamentals","Building Systems","Preventive Maintenance",
               "Health, Safety & Compliance","Contractor Management",
               "Budget & Cost Management","Emergency & Incident Management",
               "Team & FM Leadership","Final Manager Assessment"]

DB_POOL: Optional[asyncpg.Pool] = None

async def make_pool():
    kw = {}
    if "supabase" in DATABASE_URL or "sslmode=require" in DATABASE_URL:
        kw["ssl"] = sslmod.create_default_context()
    return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=8, **kw)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB_POOL
    if DATABASE_URL:
        DB_POOL = await make_pool()
    yield
    if DB_POOL:
        await DB_POOL.close()

app = FastAPI(title="FM LMS Knowledge Service", lifespan=lifespan)

@app.get("/")
async def root():
    return {"service": "FM LMS Knowledge Service", "status": "ok",
            "endpoints": ["/health", "/docs", "/ingest/upload", "/ask", "/goal",
                          "/admin/gaps"]}

# ---------------------------------------------------------------- Google Gemini
def _strip_fences(s: str) -> str:
    """Gemini occasionally wraps JSON in ``` fences even in JSON mode."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s).rstrip("`").strip()
    return s

async def llm(prompt: str, system: str = "", json_mode: bool = False, max_tokens: int = 1500) -> str:
    gen = {"temperature": 0, "maxOutputTokens": max_tokens}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    async with httpx.AsyncClient(timeout=180) as cli:
        r = await cli.post(GEMINI_BASE + "/models/" + CHAT_MODEL + ":generateContent",
                           headers={"x-goog-api-key": GEMINI_KEY}, json=body)
        r.raise_for_status()
        return _strip_fences(r.json()["candidates"][0]["content"]["parts"][0]["text"])

def _normalize(vec: list) -> list:
    """gemini-embedding-001 requires manual L2 normalization at 768 dims."""
    mag = (sum(x * x for x in vec)) ** 0.5 or 1.0
    return [x / mag for x in vec]

async def _embed_one(cli, text: str) -> list:
    for attempt in range(4):                    # backoff on free-tier 429s
        r = await cli.post(GEMINI_BASE + "/models/" + EMBED_MODEL + ":embedContent",
                           headers={"x-goog-api-key": GEMINI_KEY},
                           json={"content": {"parts": [{"text": text[:8000]}]},
                                 "outputDimensionality": EMBED_DIMS})
        if r.status_code == 429 and attempt < 3:
            await asyncio.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return _normalize(r.json()["embedding"]["values"])
    raise RuntimeError("embedding rate-limited after retries")

async def embed(texts: list) -> list:
    if not texts:
        return []
    sem = asyncio.Semaphore(5)                 # stay under free-tier RPM
    async with httpx.AsyncClient(timeout=120) as cli:
        async def one(t):
            async with sem:
                return await _embed_one(cli, t)
        return list(await asyncio.gather(*[one(t) for t in texts]))

# ---------------------------------------------------------------- zero-loss extraction
def _pdf_sections(path: str):
    from pypdf import PdfReader
    rd = PdfReader(path)
    secs, ledger = [], {}
    for i, page in enumerate(rd.pages):
        txt = page.extract_text() or ""
        ledger["paragraph"] = ledger.get("paragraph", 0) + txt.count("\n\n")
        if not txt.strip() and len(page.images) > 0:
            ledger["figure"] = ledger.get("figure", 0) + len(page.images)
            secs.append({"toc_path": "page " + str(i+1), "section": "p" + str(i+1),
                         "page": i+1, "elements": [("figure", None)],
                         "body": "", "quarantine": "image-only page: needs vision review"})
            continue
        if txt.strip():
            secs.append({"toc_path": "page " + str(i+1), "section": "p" + str(i+1),
                         "page": i+1, "elements": [("paragraph", txt)],
                         "body": txt, "quarantine": None})
    return secs, ledger

def _docx_sections(path: str):
    import docx
    d = docx.Document(path)
    paras, table_n = [], 0
    for b in d.element.body.iterchildren():
        tag = b.tag.split("}")[1]
        if tag == "p":
            paras.append(docx.text.paragraph.Paragraph(b, d).text)
        elif tag == "tbl":
            paras.append("[TABLE " + str(table_n) + "]")
            table_n += 1
    body = "\n\n".join(paras)
    secs = [{"toc_path": "document", "section": "full", "page": 0,
             "elements": [("paragraph", paras), ("table", str(table_n) + " tables")],
             "body": body, "quarantine": None}]
    return secs, {"paragraph": len(paras), "table": table_n}

def _pptx_sections(path: str):
    from pptx import Presentation
    prs = Presentation(path)
    secs, ledger = [], {"slide": 0}
    for i, slide in enumerate(prs.slides):
        texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]
        ledger["slide"] += 1
        secs.append({"toc_path": "slide " + str(i+1), "section": "slide" + str(i+1),
                     "page": i+1, "elements": [("paragraph", "\n".join(texts))],
                     "body": "\n".join(texts), "quarantine": None})
    return secs, ledger

def _txt_sections(path: str):
    with open(path, "r", errors="replace") as f:
        body = f.read()
    secs = [{"toc_path": "document", "section": "full", "page": 0,
             "elements": [("paragraph", body)], "body": body, "quarantine": None}]
    return secs, {"paragraph": 1}

async def extract_all(path: str, filename: str):
    """Zero-loss contract: every element inventoried; unparseable -> quarantine."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"
    try:
        if ext == "pdf":
            secs, ledger = await asyncio.to_thread(_pdf_sections, path)
        elif ext == "docx":
            secs, ledger = await asyncio.to_thread(_docx_sections, path)
        elif ext in ("pptx", "ppsx"):
            secs, ledger = await asyncio.to_thread(_pptx_sections, path)
        else:
            secs, ledger = await asyncio.to_thread(_txt_sections, path)
    except Exception as e:
        return {"sections": [], "ledger": {}, "quarantined": [{"kind": "file", "note": str(e)}], "coverage": 0.0}
    quarantined = [{"kind": s["elements"][0][0], "note": s["quarantine"]}
                   for s in secs if s["quarantine"]]
    total = sum(ledger.values()) or 1
    coverage = round(100.0 * (total - len(quarantined)) / total, 1)
    return {"sections": secs, "ledger": ledger, "quarantined": quarantined, "coverage": coverage}

async def split_oversized(body: str) -> list:
    """Section-coherent: split ONLY >2k-token sections at paragraph boundaries."""
    if len(body) // 4 <= CHUNK_TOK:
        return [body]
    paras, out, cur = body.split("\n\n"), [], ""
    for p in paras:
        if len(cur) // 4 > CHUNK_TOK and cur:
            out.append(cur)
            cur = cur[-int(len(cur) * OVERLAP):] + "\n\n"
        cur += p + "\n\n"
    if cur.strip():
        out.append(cur)
    return out

async def classify(name: str, sample: str):
    prompt = ("Document: " + name + "\nContent sample:\n" + sample[:1500] + "\n\n"
               + "Choose the best category AND subcategory for a Facility Management LMS.\n"
               + "Categories: " + json.dumps(CATEGORIES) + "\n"
               + 'Respond as JSON: {"category": "...", "subcategory": "..."}')
    try:
        d = json.loads(await llm(prompt, json_mode=True))
        cat = d.get("category") if d.get("category") in CATEGORIES else "Compliance"
        subs = CATEGORIES[cat]
        sub = d.get("subcategory") if d.get("subcategory") in subs else subs[0]
        return cat, sub
    except Exception:
        return "Compliance", CATEGORIES["Compliance"][0]

# ---------------------------------------------------------------- ingestion
async def run_ingestion(doc_id, path: str, filename: str):
    try:
        clam = shutil.which("clamscan")   # ClamAV when present (self-hosted); hosted skips
        if clam:
            proc = await asyncio.create_subprocess_exec(clam, "--no-summary", path)
            await proc.wait()
            if proc.returncode == 2:
                async with DB_POOL.acquire() as c:
                    await c.execute("UPDATE documents SET status='quarantined', review_note='virus detected' WHERE id=$1", doc_id)
                return
        ext = await extract_all(path, filename)
        if ext["quarantined"]:
            async with DB_POOL.acquire() as c:
                for q in ext["quarantined"]:
                    await c.execute("INSERT INTO elements (doc_id, kind, raw, quarantined, review_note) VALUES ($1,$2,'',true,$3)",
                                    doc_id, q["kind"], q["note"])
                await c.execute("UPDATE documents SET status='quarantined', coverage_pct=$2 WHERE id=$1",
                                doc_id, ext["coverage"])
            return  # publish gate: admin must clear quarantine
        cat, sub = await classify(filename, ext["sections"][0]["body"] if ext["sections"] else "")
        risk = 4 if cat in SAFETY_CATS else 2
        rows, texts, kinds_seen = [], [], set()
        for s in ext["sections"]:
            for k, _ in s["elements"]:
                kinds_seen.add(k)
            for piece in await split_oversized(s["body"]):
                if not piece.strip():
                    continue
                rows.append((doc_id, s["toc_path"], s["section"], s["page"],
                             sorted(kinds_seen), risk, piece))
                texts.append(piece)
        vectors = await embed(texts) if texts else []
        async with DB_POOL.acquire() as c:
            await c.executemany(
                """INSERT INTO chunks (doc_id, toc_path, section, page, element_kinds, risk_score, body, embedding)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                [r + (v,) for r, v in zip(rows, vectors)])
            await c.execute("""UPDATE documents SET category=$2, subcategory=$3, coverage_pct=$4,
                               toc_count=$5, status='published' WHERE id=$1""",
                            doc_id, cat, sub, ext["coverage"], len(ext["sections"]))
    except Exception as e:
        async with DB_POOL.acquire() as c:
            await c.execute("UPDATE documents SET status='error', review_note=$2 WHERE id=$1", doc_id, str(e))

@app.post("/ingest/upload")
async def upload(file: UploadFile = File(...)):
    if not DATABASE_URL:
        raise HTTPException(503, "DATABASE_URL not configured — set the Supabase connection string in deployment environment")
    if not GEMINI_KEY:
        raise HTTPException(503, "GEMINI_API_KEY not configured")
    data, total, sha = io.BytesIO(), 0, hashlib.sha256()
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        total += len(part)
        if total > MAX_MB * 1024 * 1024:
            raise HTTPException(413, "Exceeds " + str(MAX_MB) + " MB limit")
        sha.update(part)
        data.write(part)
    if total == 0:
        raise HTTPException(400, "empty file")
    doc_id = uuid.uuid4()
    os.makedirs("/tmp/uploads", exist_ok=True)
    path = "/tmp/uploads/" + str(doc_id) + "_" + file.filename
    with open(path, "wb") as f:
        f.write(data.getvalue())
    async with DB_POOL.acquire() as c:
        await c.execute("INSERT INTO documents (id, name, size_mb, sha256) VALUES ($1,$2,$3,$4)",
                        doc_id, file.filename, round(total / 1048576, 1), sha.hexdigest())
    asyncio.create_task(run_ingestion(doc_id, path, file.filename))
    return {"document_id": str(doc_id), "size_mb": round(total / 1048576, 1), "status": "processing"}

@app.get("/ingest/{doc_id}")
async def status(doc_id: uuid.UUID):
    async with DB_POOL.acquire() as c:
        row = await c.fetchrow("SELECT name, status, category, subcategory, coverage_pct, toc_count, review_note FROM documents WHERE id=$1", doc_id)
        if not row:
            raise HTTPException(404, "unknown document")
        n = await c.fetchval("SELECT count(*) FROM chunks WHERE doc_id=$1", doc_id)
    d = dict(row)
    d["section_embeddings"] = n
    return d

# ---------------------------------------------------------------- Ask FM
class AskIn(BaseModel):
    question: str
    facility: Optional[str] = None

RETRIEVE_SQL = """
SELECT c.id, c.body, c.toc_path, c.section, c.risk_score, d.name AS doc, d.category,
       1 - (c.embedding <=> $1::vector) AS sim
FROM chunks c JOIN documents d ON d.id = c.doc_id
WHERE ($2::text IS NULL OR d.category = $2)
ORDER BY c.embedding <=> $1::vector LIMIT 5
"""

STOP = {"the","a","an","of","in","to","and","or","is","are","for","on","how","do","i","what","should"}
def tokenize(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]

async def hybrid_retrieve(question, facility):
    async with DB_POOL.acquire() as c:
        qv = (await embed([question]))[0]
        sem = await c.fetch(RETRIEVE_SQL, qv, facility)
    qt = set(tokenize(question))
    scored = [{**dict(r), "score": r["sim"] + 0.02 * len(qt & set(tokenize(r["body"])))} for r in sem]
    return sorted(scored, key=lambda r: -r["score"])[:5]

ANSWER_SYSTEM = (
    "You are Ask FM, a Facility Management knowledge assistant. Answer ONLY from the provided "
    "context (RAG). Temperature 0. Cite sources for every claim. If the context lacks the answer, "
    "say so plainly. For risk_score>=4 safety content, reproduce procedure text VERBATIM.\n"
    'Output JSON: {"direct_answer": str, "procedures": [str], "warnings": [str], '
    '"sources": [{"doc": str, "section": str}], "confidence": float}'
)

@app.post("/ask")
async def ask(inp: AskIn):
    hits = await hybrid_retrieve(inp.question, inp.facility)
    if not hits:
        async with DB_POOL.acquire() as c:
            await c.execute("INSERT INTO unanswered (question) VALUES ($1)", inp.question)
        return JSONResponse({"direct_answer": "I don't have this in the knowledge base yet.",
                             "procedures": [], "warnings": [], "sources": [],
                             "logged_to_gap_dashboard": True})
    ctx = "\n\n".join("[S" + str(i+1) + "] doc=" + h["doc"] + " section=" + h["section"] + " toc=" + h["toc_path"] + " risk=" + str(h["risk_score"]) + "\n" + h["body"]
                      for i, h in enumerate(hits))
    raw = await llm("Context:\n" + ctx + "\n\nQuestion: " + inp.question, system=ANSWER_SYSTEM, json_mode=True)
    try:
        ans = json.loads(raw)
    except json.JSONDecodeError:
        ans = {"direct_answer": raw.strip(), "procedures": [], "warnings": [], "sources": []}
    if not ans.get("sources"):
        ans["sources"] = [{"doc": h["doc"], "section": h["section"]} for h in hits[:2]]
    return ans

# ---------------------------------------------------------------- Learning paths
class GoalIn(BaseModel):
    goal: str
    level: str = "Beginner"

@app.post("/goal")
async def goal(inp: GoalIn):
    async with DB_POOL.acquire() as c:
        rows = await c.fetch("SELECT category, count(*) n FROM chunks GROUP BY category")
    coverage = {r["category"]: r["n"] for r in rows}
    prompt = ("Goal: " + inp.goal + "\nStart level: " + inp.level + "\nContent coverage: " + json.dumps(coverage) + "\n"
              + "Stages: " + json.dumps(STAGE_ORDER) + "\n"
              + 'Return JSON {"stages":[{"title":str,"competency":str,"modules":[str],"content_ready":bool}]}')
    plan = json.loads(await llm(prompt, json_mode=True))
    for st in plan.get("stages", []):
        key = next((k for k in coverage if k and k.lower() in st["title"].lower()), None)
        st["content_ready"] = bool(key and coverage[key] >= 5)
    return {"path": plan}

# ---------------------------------------------------------------- admin
@app.get("/admin/gaps")
async def gaps():
    async with DB_POOL.acquire() as c:
        cat = await c.fetch("SELECT category, count(*) n FROM chunks GROUP BY category")
        un = await c.fetch("SELECT question, count(*) n FROM unanswered GROUP BY question ORDER BY n DESC LIMIT 50")
        qd = await c.fetch("SELECT id::text, name, coverage_pct FROM documents WHERE status IN ('quarantined','error')")
    return {"category_coverage": [dict(r) for r in cat],
            "unanswered_questions": [dict(r) for r in un],
            "blocked_documents": [dict(r) for r in qd]}

@app.post("/admin/quarantine/{doc_id}/clear")
async def clear_quarantine(doc_id: uuid.UUID):
    async with DB_POOL.acquire() as c:
        await c.execute("DELETE FROM elements WHERE doc_id=$1 AND quarantined", doc_id)
        name = await c.fetchval("SELECT name FROM documents WHERE id=$1", doc_id)
    if not name:
        raise HTTPException(404, "unknown document")
    path = "/tmp/uploads/" + str(doc_id) + "_" + name
    asyncio.create_task(run_ingestion(doc_id, path, name))
    return {"document_id": str(doc_id), "status": "reprocessing"}

@app.get("/health")
async def health():
    ok_db = DB_POOL is not None
    ok_key = bool(GEMINI_KEY)
    return {"ready": ok_db and ok_key, "database": ok_db, "gemini_key": ok_key,
            "max_upload_mb": MAX_MB}
