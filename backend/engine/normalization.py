import re
import difflib
from collections import defaultdict as _dd

def _toks(name: str) -> set:
    s = re.sub(r"[-_]", " ", name.lower())
    return set(re.sub(r"[^a-z0-9 ]", "", s).split())

def _seq_ratio(a: str, b: str) -> float:
    sa = " ".join(sorted(_toks(a)))
    sb = " ".join(sorted(_toks(b)))
    return difflib.SequenceMatcher(None, sa, sb).ratio()

def _tok_fuzzy_subset(a: str, b: str, thresh: float = 0.76) -> bool:
    ta = list(_toks(a))
    tb = list(_toks(b))
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    for tok in shorter:
        best = max((difflib.SequenceMatcher(None, tok, lt).ratio() for lt in longer), default=0)
        if best < thresh:
            return False
    return True

def match_score(a: str, b: str):
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0, "none"
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if shorter <= longer:
        boost = min(0.12, len(shorter) * 0.04)
        return 0.88 + boost, "subset"
    if _tok_fuzzy_subset(a, b):
        return 0.85, "typo"
    ratio = _seq_ratio(a, b)
    if ratio >= 0.88:
        return ratio, "typo"
    return 0.0, "none"

def detect_name_clusters(names: list, counts: dict) -> list[dict]:
    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            if len(_toks(pa)) >= len(_toks(pb)): parent[pb] = pa
            else: parent[pa] = pb
    multi = [n for n in names if len(_toks(n)) >= 2]
    if len(multi) <= 500:
        for i, a in enumerate(multi):
            for b in multi[i + 1:]:
                sc, why = match_score(a, b)
                if sc > 0 and why != "none": union(a, b)
    else:
        sorted_multi = sorted(multi, key=lambda x: counts.get(x, 0), reverse=True)
        top_multi = sorted_multi[:500]
        for i, a in enumerate(top_multi):
            for b in top_multi[i + 1:]:
                sc, why = match_score(a, b)
                if sc > 0 and why != "none": union(a, b)
    final = _dd(list)
    for n in names: final[find(n)].append(n)
    def best_canonical(members):
        def score(n):
            tc   = len(_toks(n))
            freq = counts.get(n, 0)
            titled  = n == n.title()
            no_pfx  = not re.match(r"^(Dr|DR)\s", n)
            return (tc, no_pfx, titled, freq, len(n))
        return max(members, key=score)
    results = []
    for root, members in final.items():
        if len(members) < 2: continue
        canon = best_canonical(members)
        variants = [m for m in members if m != canon]
        scores = [match_score(canon, v)[0] for v in variants]
        conf   = round(sum(scores) / len(scores), 3) if scores else 1.0
        ct = _toks(canon)
        suspicious = any(not (_toks(v) & ct) for v in variants)
        results.append({
            "canonical":  canon,
            "variants":   sorted(variants, key=lambda x: (-counts.get(x, 0), -len(x))),
            "confidence": conf,
            "suspicious": suspicious,
            "count":      len(members),
        })
    results.sort(key=lambda x: (-x["count"], -x["confidence"]))
    return results

_CF_PREFIX_RE = re.compile(r"^(RWA?/?|RSSB/?)\s*", re.IGNORECASE)
_CF_SEP_RE    = re.compile(r"[\s/\-]")
_CF_LZERO_RE  = re.compile(r"^(0+)(\d+)$")

def normalize_rama(x) -> str:
    s = _CF_PREFIX_RE.sub("", str(x).strip().upper())
    s = _CF_SEP_RE.sub("", s)
    m = _CF_LZERO_RE.match(s)
    return m.group(2) if m else s
