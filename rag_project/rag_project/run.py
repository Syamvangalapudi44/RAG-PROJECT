#!/usr/bin/env python3
"""
RAG over a contradictory corpus.  Run:  python run.py

Two chunkers (fixed-size/overlap vs structure-aware) x two pipelines
(naive top-3 vs conflict-aware "resolved") = 4 configurations, evaluated on
the 8 required queries.

Backends (auto-detected, override with env vars):
  Embeddings : EMBED=st   -> sentence-transformers all-MiniLM-L6-v2 (pip install sentence-transformers)
               EMBED=hash -> built-in TF-IDF-style hashing embeddings (default, zero downloads)
  Generation : LLM_PROVIDER=anthropic|openai|ollama (needs ANTHROPIC_API_KEY / OPENAI_API_KEY / local Ollama)
               LLM_MODEL=<model name>   OPENAI_BASE_URL=<any OpenAI-compatible proxy>
               If no provider is configured, a deterministic EXTRACTIVE generator is used
               (no LLM at all) so the script always runs end-to-end.  See WRITEUP.md.
  Flags      : --no-llm  force the extractive generator
"""
import os, re, sys, json, csv, zlib, datetime as dt
import urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "corpus")
OUT = os.path.join(ROOT, "results")
TODAY = dt.date.today()

# ----------------------------------------------------------------------------
# Text utils
# ----------------------------------------------------------------------------
STOP = set("""a an the of for to i am is are was were be been do does did will would can could should
me my we our you your it its in on at by with from as and or but not no if then than that this these
those what which who whom whose how when where why there here about into over under per any each every
all some such so also have has had get got""".split())
# words that carry no topical signal for the coverage guard / sentence scorer
GENERIC = set("company current currently apply applies taken trip need keep amount".split())
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def toks(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def stem(w):
    for suf in ("ing", "ed", "es", "al", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def topical_stems(t):
    """stems used for coverage + sentence scoring: no stopwords, generic words, months, numbers"""
    return [stem(w) for w in toks(t)
            if w not in STOP and w not in GENERIC and w not in MONTHS and not w.isdigit()]


def ws(t):
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------
# Ingestion: documents + metadata extraction (effective date, supersedes, amends)
# ----------------------------------------------------------------------------
DATE_RE = r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})"


def to_date(d, m, y):
    return dt.date(int(y), MONTHS.index(m.lower()) + 1, int(d))


class Doc:
    pass


def make_doc(fn, text):
    d = Doc()
    d.file, d.text = fn, text
    d.title = next(l.strip() for l in text.splitlines() if l.strip())
    d.family = re.sub(r"\s*-\s*" + DATE_RE, "", d.title).lower().strip()
    flat = ws(text)
    d.effective = None
    m = re.search(r"Effective\s+" + DATE_RE, flat)
    if m:
        d.effective = to_date(*m.groups())
    elif re.search(r"Effective immediately", flat):
        m = re.search(DATE_RE, d.title)          # memo date in the title line
        if m:
            d.effective = to_date(*m.groups())
    m = re.search(r"Supersedes the policy dated\s+" + DATE_RE, flat)
    d.supersedes_date = to_date(*m.groups()) if m else None
    # explicit clause cross-references, e.g. "clause 5.2 of the Equipment and Asset Policy is raised"
    d.amends = [(c, t.strip().lower()) for c, t in
                re.findall(r"clause\s+(\d+\.\d+)\s+of\s+the\s+([A-Za-z ]+?)\s+is\b", flat)]
    d.superseded_by = None
    d.amend_targets = []          # (clause, target Doc)
    # clause spans (char ranges) used to know which clause a fixed-size chunk covers
    starts = [(m.group(1), m.start()) for m in re.finditer(r"(?m)^(\d+\.\d+)\s", text)]
    d.clause_spans = [(cid, s, starts[i + 1][1] if i + 1 < len(starts) else len(text))
                      for i, (cid, s) in enumerate(starts)]
    return d


def load_docs():
    docs = {}
    for fn in sorted(os.listdir(CORPUS)):
        if fn.endswith(".md"):
            with open(os.path.join(CORPUS, fn), encoding="utf-8", newline="") as f:
                docs[fn] = make_doc(fn, f.read())
    link_docs(docs)
    return docs


def link_docs(docs):
    for d in docs.values():
        if d.supersedes_date:
            for o in docs.values():
                if o is not d and o.family == d.family and o.effective == d.supersedes_date:
                    o.superseded_by = d
        for clause, target in d.amends:
            for o in docs.values():
                if o is not d and (o.family == target or target in o.family):
                    d.amend_targets.append((clause, o))


def in_force(d, asof):
    if d.effective and d.effective > asof:
        return False
    if d.superseded_by and d.superseded_by.effective and d.superseded_by.effective <= asof:
        return False
    return True


# ----------------------------------------------------------------------------
# Chunking strategy 1: fixed-size character windows with overlap (structure-blind)
# ----------------------------------------------------------------------------
def mk(doc, body, prefix="", start=None, end=None, clauses=None, kind="fixed"):
    return dict(file=doc.file, body=body, prefix=prefix,
                text=(prefix + "\n" + body) if prefix else body,
                start=start, end=end, clauses=clauses or [], kind=kind)


def chunk_fixed(doc, size=300, overlap=60):
    t, step, out = doc.text, size - overlap, []
    for i in range(0, len(t), step):
        piece = t[i:i + size]
        if piece.strip():
            e = i + len(piece)
            cl = [c for c, s, en in doc.clause_spans if s < e and en > i]
            out.append(mk(doc, piece, "", i, e, cl, "fixed"))
        if i + size >= len(t):
            break
    return out


# ----------------------------------------------------------------------------
# Chunking strategy 2: structure-aware (clause / table-row / Q&A / paragraph),
# each chunk carries its document header (title, effective date, supersedes)
# ----------------------------------------------------------------------------
def block_kind(lines):
    if all(l.lstrip().startswith("|") for l in lines):
        return "table"
    if re.match(r"^\d+\.\d+\s", lines[0]):
        return "clause"
    if lines[0].startswith("Q:"):
        return "qa"
    return "plain"


def chunk_structure(doc):
    blocks = [b.split("\n") for b in re.split(r"\n\s*\n", doc.text.strip("\n")) if b.strip()]
    kinds = [block_kind(b) for b in blocks]
    first = next((i for i, k in enumerate(kinds) if k != "plain"), None)
    structured = first is not None
    ctx_end = first if structured else 1
    ctx = ws(" ".join(" ".join(b) for b in blocks[:ctx_end]))
    out = []
    for bi in range(ctx_end, len(blocks)):
        lines, kind = blocks[bi], kinds[bi]
        if kind == "table":
            hdr = [h.strip() for h in lines[0].strip().strip("|").split("|")]
            prefix = ctx + " Columns: " + " | ".join(hdr)
            for l in lines[1:]:
                if re.match(r"^\|[\s\-:|]+\|$", l.strip()):
                    continue
                cells = [c.strip() for c in l.strip().strip("|").split("|")]
                out.append(mk(doc, "; ".join(f"{h}={v}" for h, v in zip(hdr, cells)),
                              prefix, kind="table-row"))
        elif kind == "clause":
            items = []
            for l in lines:
                if re.match(r"^\d+\.\d+\s", l):
                    items.append([l.strip()])
                else:
                    items[-1].append(l.strip())
            for it in items:
                body = ws(" ".join(it))
                out.append(mk(doc, body, ctx, clauses=[body.split()[0]], kind="clause"))
        elif kind == "qa":
            out.append(mk(doc, ws(" ".join(lines)), ctx, kind="qa"))
        else:
            out.append(mk(doc, ws(" ".join(lines)), ctx if not structured else doc.title, kind="paragraph"))
    if not out:                                   # one-block document
        out.append(mk(doc, ws(doc.text), kind="paragraph"))
    return out


CHUNKERS = {"fixed": chunk_fixed, "structure": chunk_structure}

# ----------------------------------------------------------------------------
# Embeddings + vector index (plain NumPy cosine)
# ----------------------------------------------------------------------------
class HashEmb:
    name = "hash-tfidf-2048"
    DIM = 2048

    def feats(self, text):
        s = [stem(t) for t in toks(text) if t not in STOP]
        return s + [a + "_" + b for a, b in zip(s, s[1:])]

    def _h(self, f):
        return zlib.crc32(f.encode()) % self.DIM

    def fit(self, texts):
        df = np.zeros(self.DIM)
        for t in texts:
            for i in {self._h(f) for f in self.feats(t)}:
                df[i] += 1
        self.idf = np.log((len(texts) + 1) / (df + 1)) + 1

    def encode(self, texts):
        M = np.zeros((len(texts), self.DIM))
        for r, t in enumerate(texts):
            for f in self.feats(t):
                M[r, self._h(f)] += 1
        M = np.log1p(M) * self.idf
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


class STEmb:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("all-MiniLM-L6-v2")
        self.name = "sentence-transformers/all-MiniLM-L6-v2"

    def fit(self, texts):
        pass

    def encode(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True))


def make_embedder():
    if os.environ.get("EMBED", "hash") == "st":
        return STEmb()
    return HashEmb()


class Index:
    def __init__(self, chunks, emb):
        self.chunks, self.emb = chunks, emb
        emb.fit([c["text"] for c in chunks])
        self.M = emb.encode([c["text"] for c in chunks])

    def scores(self, q):
        return self.M @ self.emb.encode([q])[0]


# ----------------------------------------------------------------------------
# Query understanding
# ----------------------------------------------------------------------------
def parse_asof(q):
    m = re.search(DATE_RE, q)
    if m and m.group(2).lower() in MONTHS:
        return to_date(*m.groups())
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", q, re.I)
    if m:
        return dt.date(int(m.group(2)), MONTHS.index(m.group(1).lower()) + 1, 15)
    m = re.search(r"\b((?:19|20)\d{2})\b", q)
    if m:
        return dt.date(int(m.group(1)), 12, 31)
    return None


def coverage(q, texts):
    qs = set(topical_stems(q))
    if not qs:
        return 1.0
    have = set()
    for t in texts:
        have |= set(topical_stems(t))
    return len(qs & have) / len(qs)


COVERAGE_MIN = 0.5      # abstain if less than half the topical query terms appear in the top-3 chunks


# ----------------------------------------------------------------------------
# Generation: LLM (optional) or deterministic extractive fallback
# ----------------------------------------------------------------------------
def llm_config():
    if "--no-llm" in sys.argv:
        return None
    p = os.environ.get("LLM_PROVIDER", "")
    if not p:
        if os.environ.get("ANTHROPIC_API_KEY"):
            p = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            p = "openai"
    if not p:
        return None
    default = {"anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini", "ollama": "llama3"}[p]
    return dict(provider=p, model=os.environ.get("LLM_MODEL", default))


def call_llm(cfg, system, user):
    p, model = cfg["provider"], cfg["model"]
    if p == "anthropic":
        url, hdr = "https://api.anthropic.com/v1/messages", {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"}
        body = dict(model=model, max_tokens=500, temperature=0, system=system,
                    messages=[{"role": "user", "content": user}])
        get = lambda r: r["content"][0]["text"]
    elif p == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        url, hdr = base + "/chat/completions", {
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none"),
            "content-type": "application/json"}
        body = dict(model=model, temperature=0,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        get = lambda r: r["choices"][0]["message"]["content"]
    else:  # ollama
        url, hdr = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat", {
            "content-type": "application/json"}
        body = dict(model=model, stream=False, options={"temperature": 0},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        get = lambda r: r["message"]["content"]
    req = urllib.request.Request(url, json.dumps(body).encode(), hdr)
    with urllib.request.urlopen(req, timeout=120) as r:
        return get(json.loads(r.read())).strip()


SYS_NAIVE = ("Answer the question using ONLY the context chunks. Cite the source file name in square "
             "brackets, e.g. [travel_policy_2024.md], after every claim. If the context does not contain "
             "the answer, reply exactly: ABSTAIN: <one-line reason>.")
SYS_RESOLVED = (SYS_NAIVE + " Each chunk is labelled with a role. IN_FORCE and AMENDMENT chunks state what "
                "currently applies (or applied on the date asked). SUPERSEDED / NOT_IN_FORCE chunks are "
                "shown only for transparency: never present their values as current. State which version "
                "you used and, where relevant, mention the superseded value and its file.")


def strip_clause(c, doc, clause):
    """body of chunk c with the text of `clause` removed (used for amended clauses)"""
    if c["kind"] == "clause":
        return "" if c["clauses"] and c["clauses"][0] == clause else c["body"]
    if c["start"] is None:
        return c["body"]
    keep = list(c["body"])
    for cid, s, e in doc.clause_spans:
        if cid == clause:
            for k in range(max(s, c["start"]), min(e, c["end"])):
                keep[k - c["start"]] = ""
    return "".join(keep)


def units(body):
    cur, us = [], []
    for l in body.split("\n"):
        if re.match(r"^\s*(\d+\.\d+\s|\||Q:)", l) and cur:
            us.append(" ".join(cur)); cur = []
        cur.append(l.strip())
    if cur:
        us.append(" ".join(cur))
    out = []
    for u in us:
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", u) if s.strip()]
    return out


def pick(query, items):
    """items: [(file, body)] in rank order -> tied-best sentences from the winning file"""
    qs = set(topical_stems(query))
    scored = []
    for rank, (f, body) in enumerate(items):
        for s in units(body):
            sc = len(qs & set(topical_stems(s)))
            if sc:
                scored.append((sc, rank, f, s))
    if not scored:
        return []
    top = max(x[0] for x in scored)
    best = min((x for x in scored if x[0] == top), key=lambda x: x[1])
    seen, res = set(), []
    for sc, rank, f, s in scored:
        if sc == top and f == best[2] and s not in seen:
            seen.add(s); res.append((f, s))
    pick.top = top            # score of the winning sentences (used to filter the history disclosure)
    return res


def extractive(query, ctx, history, notes, resolved):
    """ctx/history: [{file, body}]; deterministic stand-in for an LLM"""
    ans = pick(query, [(c["file"], c["body"]) for c in ctx])
    ans_top = getattr(pick, "top", 0)
    if not ans:
        return "ABSTAIN: no sentence in the retrieved chunks matches the question."
    lines = ["Answer:"] + [f"- {s} [{f}]" for f, s in ans]
    if resolved:
        if notes:
            lines += ["Resolution:"] + [f"- {n}" for n in notes]
        used = {s for _, s in ans}
        extra = []
        for h in history:
            for f, s in pick(query, [(h["file"], h["body"])]):
                if pick.top >= ans_top and s not in used and (f, s) not in extra:
                    extra.append((f, s))
        if extra:
            lines.append("Superseded / not-in-force versions (shown for transparency, NOT current):")
            lines += [f"- (superseded, not in force) {s} [{f}]" for f, s in extra]
    return "\n".join(lines)


def generate(cfg, query, ctx, history, notes, resolved):
    if cfg is None:
        return extractive(query, ctx, history, notes, resolved)
    parts = []
    for c in ctx:
        parts.append(f"[{c['file']} | {c['role']} | effective {c['eff']}]\n{c['body']}")
    for h in history:
        parts.append(f"[{h['file']} | {h['role']} | effective {h['eff']}]\n{h['body']}")
    user = "CONTEXT:\n\n" + "\n\n".join(parts)
    if resolved and notes:
        user += "\n\nRESOLUTION NOTES (from ingestion metadata):\n- " + "\n- ".join(notes)
    user += f"\n\nQUESTION: {query}"
    return call_llm(cfg, SYS_RESOLVED if resolved else SYS_NAIVE, user)


# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------
def nums(t):
    return set(re.findall(r"\d[\d,]*", t))


def detect_conflict(ctx_items, docs):
    """in-force chunks from different files of the SAME family covering the same clause with different figures
    and no date ordering that resolves them -> unresolved conflict"""
    out = []
    for i, a in enumerate(ctx_items):
        for b in ctx_items[i + 1:]:
            da, db = docs[a["file"]], docs[b["file"]]
            if a["file"] != b["file"] and da.family == db.family and set(a["clauses"]) & set(b["clauses"]) \
                    and nums(a["body"]) != nums(b["body"]):
                out.append((a, b))
    return out


def run_naive(index, q, cfg, docs):
    sc = index.scores(q)
    top = list(np.argsort(-sc)[:3])
    ctx = [dict(index.chunks[i], role="CONTEXT", eff=str(docs[index.chunks[i]["file"]].effective)) for i in top]
    if cfg is None and sc[top[0]] < 0.05:
        return dict(answer="ABSTAIN: nothing in the corpus resembles the question.", retrieved=ctx, ctx=ctx)
    return dict(answer=generate(cfg, q, ctx, [], [], False), retrieved=ctx, ctx=ctx)


def run_resolved(index, q, cfg, docs, pool=8, final_k=3):
    chunks = index.chunks
    asof_q = parse_asof(q)
    asof = asof_q or TODAY
    sc = index.scores(q)
    order = list(np.argsort(-sc))
    top3 = order[:3]
    # 1) guard: abstain when the corpus does not cover the question's topical terms
    cov = coverage(q, [chunks[i]["text"] for i in top3])
    if cov < COVERAGE_MIN:
        return dict(answer=f"ABSTAIN: the documents do not cover this topic (coverage {cov:.2f}).",
                    retrieved=[chunks[i] for i in top3], ctx=[chunks[i] for i in top3])

    # 2) version resolution: keep only chunks of documents in force at `asof`
    kept, history = [], []
    add = lambda i, l: l.append(i) if i not in l else None
    for i in order[:pool]:
        d = docs[chunks[i]["file"]]
        if in_force(d, asof):
            add(i, kept)
        else:
            add(i, history)
            for o in docs.values():
                if o is not d and o.family == d.family and in_force(o, asof):
                    j = max((k for k, c in enumerate(chunks) if c["file"] == o.file), key=lambda k: sc[k])
                    add(j, kept)
    kept.sort(key=lambda k: -sc[k])
    final = kept[:final_k]

    # 3) amendments: memo -> clause cross-reference, applied only if effective at `asof`
    attached, strip, notes = [], {}, []
    for A in docs.values():
        if not A.effective or A.effective > asof:
            continue
        for clause, T in A.amend_targets:
            touched = any(chunks[i]["file"] == A.file for i in final) or \
                any(chunks[i]["file"] == T.file and clause in chunks[i]["clauses"] for i in final)
            if not touched:
                continue
            for k, c in enumerate(chunks):
                if c["file"] == A.file:
                    add(k, attached)
                if c["file"] == T.file and clause in c["clauses"]:
                    strip.setdefault(k, set()).add(clause)
                    add(k, history)
            notes.append(f"{A.file} (effective {A.effective}) amends clause {clause} of {T.file}; "
                         f"the amended figure applies and the original clause is superseded.")
    ctx_idx = [k for k in attached] + [k for k in final if k not in attached]

    # 4) version notes (no silent picks)
    fams = {docs[chunks[i]["file"]].family for i in ctx_idx}
    for d in docs.values():
        if d.family in fams and d.superseded_by:
            sup = d.superseded_by
            state = "in force" if in_force(sup, asof) else "not yet in force"
            notes.append(f"{sup.file} (effective {sup.effective}) supersedes {d.file} "
                         f"(effective {d.effective}); on {asof}{' (date taken from the question)' if asof_q else ''} "
                         f"{sup.file if in_force(sup, asof) else d.file} is the version in force.")

    def view(k, role):
        c = chunks[k]
        body = c["body"]
        for cl in strip.get(k, ()):
            body = strip_clause(c, docs[c["file"]], cl)
        return dict(c, body=body, role=role, eff=str(docs[c["file"]].effective))

    ctx = [view(k, "AMENDMENT" if k in attached else "IN_FORCE") for k in ctx_idx]
    hist = []
    for k in history:
        if k in ctx_idx and k not in strip:
            continue
        d = docs[chunks[k]["file"]]
        role = "SUPERSEDED" if (d.superseded_by and in_force(d.superseded_by, asof)) or k in strip else "NOT_IN_FORCE"
        h = dict(chunks[k], role=role, eff=str(d.effective))
        hist.append(h)
    # a chunk that is both ctx (stripped) and history: history shows the full original clause text
    # 5) unresolved-conflict handling: refuse and surface both
    conf = detect_conflict(ctx, docs)
    if conf:
        a, b = conf[0]
        msg = (f"SOURCES CONFLICT - cannot determine which applies: {a['file']} says \"{ws(a['body'])}\"; "
               f"{b['file']} says \"{ws(b['body'])}\". Neither is superseded by date metadata. "
               f"Please confirm with the policy owner.")
        return dict(answer=msg, retrieved=ctx, ctx=ctx, conflict=True)
    ans = generate(cfg, q, [c for c in ctx if c["body"].strip()], hist, notes, True)
    return dict(answer=ans, retrieved=ctx, ctx=ctx)


# ----------------------------------------------------------------------------
# Gold standard + automatic grading (heuristic; answers are saved for manual audit)
# ----------------------------------------------------------------------------
P = lambda s: r"(?<![\d,])" + s + r"(?!\d)"
T6, T9 = P(r"6,?500"), P(r"9,?000")
GOLD = [
    dict(id=1, q="What is the international travel per-diem?", abstain=False,
         files={"travel_policy_2024.md"}, inc=[T9], exc=[T6],
         ev_files={"travel_policy_2024.md"}, ev=r"International travel per-diem is INR 9,000"),
    dict(id=2, q="What is the current limit for home office equipment?", abstain=False,
         files={"memo_2024_08.md"}, inc=[P(r"40,?000")], exc=[P(r"25,?000")],
         ev_files={"memo_2024_08.md"}, ev=r"40,000"),
    dict(id=3, q="I am a contractor. Will the company buy me a laptop?", abstain=False, neg=True,
         files={"equipment_policy.md"}, inc=[r"not covered|supply their own|own equipment|exclud|not eligible"], exc=[],
         ev_files={"equipment_policy.md"}, ev=r"not covered by this policy"),
    dict(id=4, q="What notice period applies to a Lead?", abstain=False,
         files={"notice_periods.md"}, inc=[P(r"90")], exc=[r"\b30 days\b", r"\b60 days\b"],
         ev_files={"notice_periods.md"}, ev=r"Lead.*90"),
    dict(id=5, q="Above what amount do I need to keep a receipt?", abstain=False,
         files={"travel_policy_2024.md"}, inc=[P(r"2,?500")], exc=[P(r"1,?000")],
         ev_files={"travel_policy_2024.md"}, ev=r"Receipts are required.*2,500"),
    dict(id=6, q="What is the per-diem for a trip taken in June 2019?", abstain=False,
         files={"travel_policy_2019.md"}, inc=[P(r"2,?000"), T6], exc=[P(r"3,?200"), T9],
         ev_files={"travel_policy_2019.md"}, ev=r"per-diem is INR"),
    dict(id=7, q="Who approves business class travel?", abstain=False,
         files={"travel_policy_2024.md", "travel_policy_2019.md"}, inc=[r"Director"], exc=[],
         ev_files={"travel_policy_2024.md", "travel_policy_2019.md"}, ev=r"Business class requires\s+Director"),
    dict(id=8, q="What is the company's parental leave entitlement?", abstain=True,
         files=set(), inc=[], exc=[], ev_files=set(), ev=""),
]
HIST = re.compile(r"supersed|previous|earlier|prior|no longer|not in force|former|original|\bwas\b|\bold\b", re.I)
KNOWN = None


def cited_files(ans, docs):
    return {f for f in docs if f in ans}


def violations(ans, pats):
    segs = re.split(r"\n|(?<=[.;])\s+", ans)
    return [p for p in pats for s in segs if re.search(p, s) and not HIST.search(s)]


def relevant(c, g):
    return c["file"] in g["ev_files"] and re.search(g["ev"], c["body"], re.S | re.I) is not None


def grade(g, res, docs):
    ans = res["answer"]
    abstained = ans.startswith("ABSTAIN")
    cited = set() if abstained else cited_files(ans, docs)
    retrieved = res["retrieved"]
    if g["abstain"]:
        return dict(abstained=abstained, cited=sorted(cited), correct=abstained,
                    failure=None if abstained else "fabricated")
    if abstained:
        return dict(abstained=True, cited=[], correct=False, failure="abstained-wrongly")
    bad = violations(ans, g["exc"])
    missing = [p for p in g["inc"] if not re.search(p, ans, re.I)]
    ok_cite = bool(cited & g["files"])
    if not bad and not missing and ok_cite:
        return dict(abstained=False, cited=sorted(cited), correct=True, failure=None)
    ev_seen = any(relevant(c, g) for c in retrieved)
    if res.get("conflict"):
        f = "conflict-refused"
    elif bad:
        f = "conflict" if ev_seen else "stale"
    elif missing and g.get("neg") and re.search(r"laptop|provid", ans, re.I):
        f = "negation"
    elif missing and any(c["file"] in g["files"] for c in retrieved):
        f = "fragment"
    elif missing:
        f = "retrieval-miss"
    else:
        f = "miscited"
    return dict(abstained=False, cited=sorted(cited), correct=False, failure=f)


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
def selftest_conflict():
    """two UNDATED docs of the same family disagreeing -> system must refuse and show both"""
    a = make_doc("pol_a.md", "LEAVE POLICY\n\n1.1 Casual leave is 10 days per year.\n")
    b = make_doc("pol_b.md", "LEAVE POLICY\n\n1.1 Casual leave is 12 days per year.\n")
    docs = {"pol_a.md": a, "pol_b.md": b}
    link_docs(docs)
    ctx = [chunk_structure(a)[0], chunk_structure(b)[0]]
    assert detect_conflict(ctx, docs), "conflict detector failed"
    return True


def main():
    docs = load_docs()
    cfg = llm_config()
    emb_name = make_embedder().name
    gen_name = f"LLM {cfg['provider']}:{cfg['model']}" if cfg else "extractive (no LLM)"
    print(f"embeddings: {emb_name} | generation: {gen_name} | today={TODAY}")
    print("conflict-path self-test:", "PASS" if selftest_conflict() else "FAIL")
    os.makedirs(OUT, exist_ok=True)

    runs, all_rows, chunk_sizes = {}, [], {}
    for cname, cfn in CHUNKERS.items():
        chunks = [c for d in docs.values() for c in cfn(d)]
        for i, c in enumerate(chunks):
            c["id"] = f"{cname}-{i}"
        chunk_sizes[cname] = [len(c["text"]) for c in chunks]
        index = Index(chunks, make_embedder())
        for pname, fn in (("naive", run_naive), ("resolved", run_resolved)):
            rows = []
            for g in GOLD:
                res = fn(index, g["q"], cfg, docs) if pname == "naive" else fn(index, g["q"], cfg, docs)
                gr = grade(g, res, docs)
                top3 = res["retrieved"][:3]
                rel = sum(relevant(c, g) for c in top3)
                rows.append(dict(chunking=cname, pipeline=pname, qid=g["id"], query=g["q"],
                                 answer=res["answer"], retrieved=[c["id"] for c in res["retrieved"]],
                                 rel_in_top3=rel, **gr))
            runs[(cname, pname)] = (chunks, rows)
            all_rows += rows

    # ---- metrics
    summary = []
    for (cname, pname), (chunks, rows) in runs.items():
        n = len(rows)
        ret_rows = [r for r in rows if r["qid"] != 8]
        fails = {}
        for r in rows:
            if r["failure"]:
                fails[r["failure"]] = fails.get(r["failure"], 0) + 1
        summary.append(dict(
            chunking=cname, pipeline=pname,
            accuracy=sum(r["correct"] for r in rows) / n, correct=sum(r["correct"] for r in rows),
            precision_at_3=sum(r["rel_in_top3"] for r in ret_rows) / (3 * len(ret_rows)),
            hit_at_3=sum(r["rel_in_top3"] > 0 for r in ret_rows) / len(ret_rows),
            n_chunks=len(chunks), mean_chunk_chars=float(np.mean([len(c["text"]) for c in chunks])),
            mean_body_chars=float(np.mean([len(c["body"]) for c in chunks])),
            abstention_rate=sum(r["abstained"] for r in rows) / n, failures=fails))

    # ---- write outputs
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(dict(embeddings=emb_name, generation=gen_name, summary=summary, rows=all_rows), f, indent=2, default=str)
    with open(os.path.join(OUT, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunking", "pipeline", "qid", "query", "answer", "cited", "correct", "failure"])
        for r in all_rows:
            w.writerow([r["chunking"], r["pipeline"], r["qid"], r["query"], r["answer"],
                        ";".join(r["cited"]), r["correct"], r["failure"] or ""])
    md = [f"Embeddings: `{emb_name}` | Generation: `{gen_name}`\n",
          "| chunking | pipeline | accuracy | P@3 | hit@3 | #chunks | mean chunk chars | abstention rate | failures |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['chunking']} | {s['pipeline']} | {s['correct']}/8 = {s['accuracy']:.0%} | "
                  f"{s['precision_at_3']:.2f} | {s['hit_at_3']:.2f} | {s['n_chunks']} | {s['mean_chunk_chars']:.0f} | "
                  f"{s['abstention_rate']:.0%} | {s['failures'] or '-'} |")
    md += ["", "### Per-query results", "",
           "| chunking | pipeline | Q | correct | failure | cited | answer |", "|---|---|---|---|---|---|---|"]
    for r in all_rows:
        a = r["answer"].replace("\n", " ⏎ ").replace("|", "\\|")
        md.append(f"| {r['chunking']} | {r['pipeline']} | {r['qid']} | {'Y' if r['correct'] else 'N'} | "
                  f"{r['failure'] or ''} | {', '.join(r['cited'])} | {a} |")
    open(os.path.join(OUT, "comparison.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:2 + 2 + len(summary)]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        mx = max(max(v) for v in chunk_sizes.values())
        bins = np.linspace(0, mx + 20, 16)
        for a, (name, v) in zip(ax, chunk_sizes.items()):
            a.hist(v, bins=bins, color="#4C72B0" if name == "fixed" else "#DD8452", edgecolor="k")
            a.axvline(np.mean(v), color="r", ls="--", label=f"mean {np.mean(v):.0f}")
            a.set_title(f"{name}: n={len(v)}, min={min(v)}, max={max(v)}")
            a.set_xlabel("chunk length (characters, text that is embedded)")
            a.set_ylabel("# chunks")
            a.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "chunk_size_distribution.png"), dpi=130)
    except ImportError:
        print("matplotlib not installed: skipped plot (pip install matplotlib)")
    print(f"\nfull results in {OUT}/ (comparison.md, per_query.csv, summary.json, chunk_size_distribution.png)")


if __name__ == "__main__":
    main()
