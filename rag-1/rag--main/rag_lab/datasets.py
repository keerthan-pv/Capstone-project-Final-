"""Dataset layer.

* ``make_fixture``       — deterministic OFFLINE corpus for tests/demos. NOT a research benchmark.
* ``load_squad_v2``      — SQuAD 2.0 from Hugging Face (answer spans + genuinely unanswerable questions).
* ``load_msmarco``       — MS MARCO v2.1 from Hugging Face (human relevance labels + no-answer queries).
* ``load_beir``          — BEIR datasets (e.g. scifact, nfcorpus) for multi-domain generalisation.

The row→dataset converters are pure functions so they are unit-tested against the exact
Hugging Face schemas without needing network access.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import random
from typing import Any, Dict, Iterable, List, Optional

from .textutil import content_id
from .types import EvalDataset, Passage, Query

log = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lab")


# ───────────────────────────── persistence ─────────────────────────────────
def save_dataset(ds: EvalDataset, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"name": ds.name, "version": ds.version, "notes": ds.notes,
               "passages": [p.__dict__ for p in ds.passages], "queries": [q.__dict__ for q in ds.queries]}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def load_dataset_file(path: str) -> EvalDataset:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return EvalDataset(name=d["name"], version=d.get("version", "1"), notes=d.get("notes", ""),
                       passages=[Passage(**p) for p in d["passages"]], queries=[Query(**q) for q in d["queries"]])


# ───────────────────────────── pure converters ─────────────────────────────
def squad_rows_to_dataset(rows: Iterable[Dict[str, Any]], name: str = "squad_v2",
                          max_queries: Optional[int] = None) -> EvalDataset:
    """rows follow rajpurkar/squad_v2: id, title, context, question, answers{text[], answer_start[]}."""
    passages: Dict[str, Passage] = {}
    queries: List[Query] = []
    for row in rows:
        ctx = (row.get("context") or "").strip()
        if not ctx:
            continue
        pid = "sq_" + hashlib.sha1(ctx.encode()).hexdigest()[:12]
        passages.setdefault(pid, Passage(pid, ctx, title=row.get("title", "")))
        texts = list((row.get("answers") or {}).get("text") or [])
        answerable = len(texts) > 0
        queries.append(Query(
            qid=str(row.get("id")), question=(row.get("question") or "").strip(), answers=texts,
            relevant={pid: 1} if answerable else {},           # unanswerable ⇒ no relevant passage
            answerable=answerable, qtype="factual" if answerable else "unanswerable", source="real",
            meta={"context_pid": pid}))
        if max_queries and len(queries) >= max_queries:
            break
    return EvalDataset(name=name, passages=list(passages.values()), queries=queries,
                       notes="SQuAD 2.0: unanswerable questions have no relevant passage; context_pid keeps the plausible-but-wrong paragraph.")


def msmarco_rows_to_dataset(rows: Iterable[Dict[str, Any]], name: str = "msmarco_v2.1",
                            max_queries: Optional[int] = None, max_unanswerable_frac: float = 0.15) -> EvalDataset:
    """rows follow microsoft/ms_marco v2.1: query_id, query, query_type, answers[], passages{is_selected[], passage_text[], url[]}."""
    passages: Dict[str, Passage] = {}
    queries: List[Query] = []
    n_unans = 0
    for row in rows:
        q = (row.get("query") or "").strip()
        pdata = row.get("passages") or {}
        texts, sel = pdata.get("passage_text") or [], pdata.get("is_selected") or []
        answers = [a for a in (row.get("answers") or []) if a and a.strip() and a != "No Answer Present."]
        relevant: Dict[str, int] = {}
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if len(t) < 30:
                continue
            pid = "ms_" + content_id(t)
            passages.setdefault(pid, Passage(pid, t, meta={"url": (pdata.get("url") or [""] * len(texts))[i]}))
            if i < len(sel) and sel[i]:
                relevant[pid] = 1
        if not q or not texts:
            continue
        if answers and relevant:
            ans = True
        elif not answers and n_unans < max_unanswerable_frac * max(10, (max_queries or 1000)):
            ans, relevant = False, {}
            n_unans += 1
        else:
            continue
        queries.append(Query(qid=str(row.get("query_id", len(queries))), question=q, answers=answers if ans else [],
                             relevant=relevant, answerable=ans, qtype=str(row.get("query_type", "unknown")).lower(),
                             source="real"))
        if max_queries and len(queries) >= max_queries:
            break
    return EvalDataset(name=name, passages=list(passages.values()), queries=queries,
                       notes="MS MARCO v2.1: relevance = is_selected; 'No Answer Present.' queries are treated as unanswerable.")


def beir_to_dataset(corpus_rows, query_rows, qrels_rows, name: str, max_queries: Optional[int] = None) -> EvalDataset:
    """BeIR/<name> (corpus: _id,title,text; queries: _id,text) + BeIR/<name>-qrels (query-id, corpus-id, score)."""
    passages = [Passage(str(r["_id"]), (r.get("text") or ""), title=r.get("title") or "") for r in corpus_rows]
    qtext = {str(r["_id"]): r["text"] for r in query_rows}
    rel: Dict[str, Dict[str, int]] = {}
    for r in qrels_rows:
        if int(r["score"]) > 0:
            rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    queries = [Query(qid=qid, question=qtext[qid], relevant=r, answerable=True, qtype="beir", source="real",
                     meta={"retrieval_only": True}) for qid, r in rel.items() if qid in qtext]
    if max_queries:
        queries = queries[:max_queries]
    return EvalDataset(name=name, passages=passages, queries=queries, notes="BEIR retrieval-only (no reference answers).")


# ───────────────────────────── Hugging Face loaders ────────────────────────
def _need_datasets():
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset
    except ImportError as e:                                         # pragma: no cover
        raise RuntimeError("`pip install datasets` and make sure you have internet access "
                           "(or load a cached file with load_dataset_file).") from e


def load_squad_v2(split: str = "validation", max_queries: int = 2000, cache_dir: str = DEFAULT_CACHE_DIR,
                  refresh: bool = False) -> EvalDataset:
    path = os.path.join(cache_dir, f"squad_v2_{split}_{max_queries}.json.gz")
    if os.path.exists(path) and not refresh:
        return load_dataset_file(path)
    ld = _need_datasets()
    ds = squad_rows_to_dataset(ld("rajpurkar/squad_v2", split=split), "squad_v2", max_queries)
    save_dataset(ds, path); log.info("cached %s", path)
    return ds


def load_msmarco(split: str = "validation", max_queries: int = 2000, cache_dir: str = DEFAULT_CACHE_DIR,
                 refresh: bool = False) -> EvalDataset:
    path = os.path.join(cache_dir, f"msmarco_v2.1_{split}_{max_queries}.json.gz")
    if os.path.exists(path) and not refresh:
        return load_dataset_file(path)
    ld = _need_datasets()
    stream = ld("microsoft/ms_marco", "v2.1", split=split, streaming=True)     # streaming: no full download
    ds = msmarco_rows_to_dataset(stream, "msmarco_v2.1", max_queries)
    save_dataset(ds, path); log.info("cached %s", path)
    return ds


def load_beir(name: str = "scifact", max_queries: int = 500, cache_dir: str = DEFAULT_CACHE_DIR,
              refresh: bool = False) -> EvalDataset:
    path = os.path.join(cache_dir, f"beir_{name}_{max_queries}.json.gz")
    if os.path.exists(path) and not refresh:
        return load_dataset_file(path)
    ld = _need_datasets()
    corpus = ld(f"BeIR/{name}", "corpus", split="corpus")
    queries = ld(f"BeIR/{name}", "queries", split="queries")
    qrels = ld(f"BeIR/{name}-qrels", split="test")
    ds = beir_to_dataset(corpus, queries, qrels, f"beir_{name}", max_queries)
    save_dataset(ds, path)
    return ds


# ───────────────────────────── offline fixture ─────────────────────────────
_SYL1 = ["Kar", "Vel", "Mir", "Tan", "Sol", "Ren", "Dor", "Hal", "Zen", "Lor", "Bra", "Quin", "Fen", "Gal", "Ost"]
_SYL2 = ["an", "is", "or", "ek", "ia", "um", "eth", "ov", "ar", "in"]
_CITY2 = ["ford", "mouth", "stad", "vale", "port", "haven", "burg", "wick"]
_THING1 = ["Vantor", "Quill", "Lumen", "Cobalt", "Drift", "Ember", "Nimbus", "Orbit", "Prism", "Tessel"]
_THING2 = ["Lens", "Resonator", "Valve", "Compass", "Loom", "Relay", "Engine", "Beacon", "Filter", "Dial"]
_FILL = [
    "Surviving records from this period are incomplete and sometimes contradict one another.",
    "Local historians have debated the exact sequence of events for several decades.",
    "A regional newspaper covered the story at the time, although coverage was brief.",
    "The archive describes the surrounding district as quiet and largely agricultural.",
    "Several contemporaries described the work as ambitious but poorly funded.",
    "Later commentators have revised earlier assessments in light of newly catalogued letters.",
    "The museum holds a small collection of related photographs and ledgers.",
    "Visitors to the area often remark on the unusually mild weather in autumn.",
    "Trade in the region depended heavily on seasonal river traffic.",
    "A commemorative plaque was installed during a restoration project.",
]


def make_fixture(n_people: int = 60, n_companies: int = 40, n_cities: int = 40, seed: int = 7,
                 n_queries: int = 240) -> EvalDataset:
    """Deterministic offline corpus of fictional entities with five query families.

    The corpus is synthetic and intentionally small: it exists so the *pipeline* (metrics,
    diagnosis, routing, caching, reports) can be exercised end-to-end offline. Numbers produced
    on it are NOT research results — run `scripts/fetch_datasets.py` for real benchmarks.
    """
    rnd = random.Random(seed)
    used: set = set()

    def uniq(make):
        for _ in range(1000):
            v = make()
            if v not in used:
                used.add(v); return v
        raise RuntimeError("name space exhausted")

    person = lambda: uniq(lambda: f"{rnd.choice(_SYL1)}{rnd.choice(_SYL2)} {rnd.choice(_SYL1)}{rnd.choice(_SYL2)}")  # noqa: E731
    city = lambda: uniq(lambda: f"{rnd.choice(_SYL1)}{rnd.choice(_CITY2)}")                                          # noqa: E731
    thing = lambda: uniq(lambda: f"{rnd.choice(_THING1)} {rnd.choice(_THING2)}")                                       # noqa: E731
    company = lambda: uniq(lambda: f"{rnd.choice(_SYL1)}{rnd.choice(_SYL2)} {rnd.choice(['Works', 'Labs', 'Industries', 'Group'])}")  # noqa: E731

    cities = [city() for _ in range(n_cities)]
    people = [{"name": person(), "born": rnd.randint(1790, 1960), "city": rnd.choice(cities)} for _ in range(n_people)]
    comps = []
    for _ in range(n_companies):
        f = rnd.choice(people)
        comps.append({"name": company(), "founded": rnd.randint(1820, 1995), "founder": f,
                      "product": thing(), "code": f"{rnd.choice('KXQZ')}{rnd.choice('KXQZ')}-{rnd.randint(1000, 9999)}"})

    def pad(facts: List[str]) -> str:
        sents = facts + rnd.sample(_FILL, k=rnd.randint(4, 7))
        rnd.shuffle(sents)
        return " ".join(sents)

    passages: List[Passage] = []
    pid_of: Dict[str, str] = {}
    for i, c in enumerate(cities):
        pid = f"city_{i}"; pid_of[c] = pid
        passages.append(Passage(pid, pad([f"{c} is a river town known for its covered market and its observatory."]), title=c))
    for i, p in enumerate(people):
        pid = f"person_{i}"; pid_of[p["name"]] = pid
        passages.append(Passage(pid, pad([f"{p['name']} was born in {p['born']} in {p['city']}.",
                                          f"{p['name']} trained as an engineer before turning to research."]), title=p["name"]))
    for i, c in enumerate(comps):
        pid = f"company_{i}"; pid_of[c["name"]] = pid
        passages.append(Passage(pid, pad([f"{c['name']} was founded in {c['founded']} by {c['founder']['name']}.",
                                          f"The {c['product']} carries product code {c['code']}."]), title=c["name"]))

    qs: List[Query] = []
    n = 0

    def add(q, ans, rel, qtype, answerable=True, para=(), **meta):
        nonlocal n
        n += 1
        qs.append(Query(qid=f"fx{n:04d}", question=q, answers=list(ans), relevant={k: 1 for k in rel},
                        answerable=answerable, qtype=qtype, paraphrases=list(para), meta=meta))

    per = n_queries // 6
    for p in rnd.sample(people, min(per, len(people))):
        add(f"When was {p['name']} born?", [str(p["born"])], [pid_of[p["name"]]], "factual",
            para=[f"What year was {p['name']} born?", f"In which year was {p['name']} born?"])
    for c in rnd.sample(comps, min(per, len(comps))):
        add(f"Which product carries product code {c['code']}?", [c["product"]], [pid_of[c["name"]]], "keyword",
            para=[f"What product has product code {c['code']}?"])
    for c in rnd.sample(comps, min(per, len(comps))):
        add(f"During which year did {c['name']} come into existence?", [str(c["founded"])], [pid_of[c["name"]]], "paraphrase",
            para=[f"In what year was {c['name']} founded?"])
    for c in rnd.sample(comps, min(per, len(comps))):
        f = c["founder"]
        add(f"In which city was the founder of {c['name']} born?", [f["city"]],
            [pid_of[c["name"]], pid_of[f["name"]]], "multihop",
            para=[f"Where was the person who founded {c['name']} born?"])
    for c in rnd.sample(comps, min(per // 2, len(comps))):
        add(f"How many employees does {c['name']} have?", [], [], "unanswerable", False, subtype="attribute_missing",
            para=[f"What is the employee count of {c['name']}?"])
    for _ in range(per - per // 2):
        ghost = uniq(lambda: f"{rnd.choice(_SYL1)}{rnd.choice(_SYL2)} {rnd.choice(_SYL1)}{rnd.choice(_SYL2)}")
        add(f"When was {ghost} born?", [], [], "unanswerable", False, subtype="entity_absent")
    for p in rnd.sample(people, min(n_queries - len(qs), len(people))):
        add(f"Who was born in {p['born']} in {p['city']}?", [p["name"]], [pid_of[p["name"]]], "factual",
            para=[f"Which person was born in {p['city']} in {p['born']}?"])
    return EvalDataset(name="offline_fixture", passages=passages, queries=qs, version=f"seed{seed}",
                       notes="Synthetic offline fixture for pipeline verification — NOT a research benchmark.")
