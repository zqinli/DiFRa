import json, re
from json_repair import repair_json

QA_RE = re.compile(
    r'"question"\s*:\s*"((?:\\.|[^"\\\x00-\x1F])*)"\s*,\s*'
    r'"answer"\s*:\s*"((?:\\.|[^"\\\x00-\x1F])*)"', re.IGNORECASE
)

BRACED_QA_RE = re.compile(
    r'\{\s*"((?:\\.|[^"\\\x00-\x1F])*)"\s*,\s*"((?:\\.|[^"\\\x00-\x1F])*)"\s*\}'
)

_CTRL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

def _json_unescape(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = _CTRL_RE.sub('', s)
    if '\\' not in s:
        return s
    s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
    try:
        return json.loads(f'"{s2}"')
    except Exception:
        return s

def _dedup(pairs):
    seen, out = set(), []
    for p in pairs:
        key = (p["question"], p["answer"])
        if key not in seen:
            seen.add(key); out.append(p)
    return out

def _first_other_str(d: dict, exclude: set[str]) -> str | None:
    for k, v in d.items():
        if k in exclude:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _append_unique(pairs, seen, q, a, max_pairs):
    q = (q or "").strip()
    a = (a or "").strip()
    if not q or not a:
        return False
    key = (q, a)
    if key in seen:
        return False
    seen.add(key)
    pairs.append({"question": q, "answer": a})
    return max_pairs is not None and len(pairs) >= max_pairs

def _extract_from_text(text: str, max_pairs: int | None = None) -> list[dict]:
    text = text or ""
    pairs, seen = [], set()

    for q, a in QA_RE.findall(text):
        try:
            qd, ad = _json_unescape(q), _json_unescape(a)
        except Exception:
            continue
        if _append_unique(pairs, seen, qd, ad, max_pairs):
            return pairs

    for q, a in BRACED_QA_RE.findall(text):
        try:
            qd, ad = _json_unescape(q), _json_unescape(a)
        except Exception:
            continue
        if _append_unique(pairs, seen, qd, ad, max_pairs):
            return pairs

    return pairs


def _extract_from_obj(obj, max_pairs: int | None = None) -> list[dict]:
    pairs, seen = [], set()

    def handle_one(d: dict):
        nonlocal pairs, seen
        q = d.get("question")
        a = d.get("answer")
        if isinstance(q, str) and isinstance(a, str):
            return _append_unique(pairs, seen, q, a, max_pairs)
        if isinstance(q, str) and a is None:
            cand = _first_other_str(d, {"question"})
            if cand:
                return _append_unique(pairs, seen, q, cand, max_pairs)
        if isinstance(a, str) and q is None:
            cand = _first_other_str(d, {"answer"})
            if cand:
                return _append_unique(pairs, seen, cand, a, max_pairs)
        return False

    if isinstance(obj, dict):
        handle_one(obj)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict) and handle_one(it):
                break
    return pairs

def safe_parse_json(val, max_pairs: int | None = None) -> list[dict]:
    if isinstance(val, (dict, list)):
        return _extract_from_obj(val, max_pairs)

    s = (val or "").strip() if isinstance(val, str) else ""
    if not s:
        return []

    try:
        obj_pairs = _extract_from_obj(json.loads(s), max_pairs)
        if obj_pairs:
            return obj_pairs
    except Exception:
        pass

    txt_pairs = _extract_from_text(s, max_pairs)
    if txt_pairs:
        return txt_pairs

    try:
        repaired = repair_json(s)
        try:
            obj_pairs = _extract_from_obj(json.loads(repaired), max_pairs)
            if obj_pairs:
                return obj_pairs
        except Exception:
            pass
        txt_pairs = _extract_from_text(repaired, max_pairs)
        if txt_pairs:
            return txt_pairs
    except Exception:
        pass

    return []