"""
Utility functions for PharmaScan application.

This module contains helper functions for:
- Name normalization and clustering
- Column profiling and mapping
- Data formatting
"""

import re
from collections import defaultdict
from typing import Optional, Union

import difflib


def fmt_number(n: Union[int, float]) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:,.0f}"


def _toks(name: str) -> set:
    """Lowercase alpha-numeric tokens from a name string."""
    return set(re.sub(r"[^a-z0-9 ]", "", name.lower()).split())


def _seq_ratio(a: str, b: str) -> float:
    """Calculate sequence similarity ratio between two normalized names."""
    sa = " ".join(sorted(_toks(a)))
    sb = " ".join(sorted(_toks(b)))
    return difflib.SequenceMatcher(None, sa, sb).ratio()


def _tok_fuzzy_subset(a: str, b: str, thresh: float = 0.76) -> bool:
    """
    True if every token in the SHORTER name has a fuzzy-close counterpart
    in the LONGER name (catches 'Aurbain'/'Urbain', 'Constatin'/'Constantin').
    """
    ta = list(_toks(a))
    tb = list(_toks(b))
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    for tok in shorter:
        best = max(
            (difflib.SequenceMatcher(None, tok, lt).ratio() for lt in longer),
            default=0,
        )
        if best < thresh:
            return False
    return True


def _match_score(a: str, b: str) -> tuple[float, str]:
    """
    Calculate match score between two names.
    
    Returns:
        Tuple of (score 0–1, reason str) where reason ∈ {'subset', 'typo', 'none'}
    """
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0, "none"

    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)

    # Rule 1 — exact token subset ('ZACHEE' ⊂ 'Niyonsenga Zachee')
    if shorter <= longer:
        boost = min(0.12, len(shorter) * 0.04)
        return 0.88 + boost, "subset"

    # Rule 2 — fuzzy-token subset: every short token ≈ some long token
    if _tok_fuzzy_subset(a, b):
        return 0.85, "typo"

    # Rule 3 — high overall char-sequence similarity
    ratio = _seq_ratio(a, b)
    if ratio >= 0.88:
        return ratio, "typo"

    return 0.0, "none"


def detect_name_clusters(names: list, counts: dict) -> list[dict]:
    """
    Cluster similar names using Union-Find algorithm.
    
    Args:
        names: List of name strings to cluster
        counts: Dictionary mapping names to their frequency counts
        
    Returns:
        List of dicts with keys: canonical, variants, method, confidence
        Only clusters with ≥2 members are returned.
    """
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            if len(_toks(pa)) >= len(_toks(pb)):
                parent[pb] = pa
            else:
                parent[pa] = pb

    # ── Pass 1: merge multi-token names (typos + reordering) ─────────────────
    multi = [n for n in names if len(_toks(n)) >= 2]
    n_multi = len(multi)

    # Optimization: skip O(n²) comparison if too many names
    if n_multi <= 500:
        for i, a in enumerate(multi):
            for b in multi[i + 1 :]:
                sc, why = _match_score(a, b)
                if sc > 0 and why != "none":
                    union(a, b)
    else:
        # For large datasets, use sampling or frequency-based filtering
        sorted_multi = sorted(multi, key=lambda x: counts.get(x, 0), reverse=True)
        top_multi = sorted_multi[:500]
        for i, a in enumerate(top_multi):
            for b in top_multi[i + 1 :]:
                sc, why = _match_score(a, b)
                if sc > 0 and why != "none":
                    union(a, b)

    # ── Pass 2: single-token names → merge only if token is unique to 1 cluster ──
    def get_clusters():
        c = defaultdict(list)
        for n in names:
            c[find(n)].append(n)
        return c

    cls1 = get_clusters()
    tok_to_roots: dict = defaultdict(set)
    for root, members in cls1.items():
        if len(members) > 1:
            for m in members:
                for t in _toks(m):
                    tok_to_roots[t].add(root)

    for name in names:
        if len(_toks(name)) != 1:
            continue
        tok = next(iter(_toks(name)))
        candidates = tok_to_roots.get(tok, set()) - {find(name)}
        if len(candidates) == 1:
            union(name, next(iter(candidates)))

    # ── Build final clusters ─────────────────────────────────────────────────
    final: dict = defaultdict(list)
    for n in names:
        final[find(n)].append(n)

    def best_canonical(members):
        def score(n):
            tc = len(_toks(n))
            freq = counts.get(n, 0)
            # Title-case preferred; "Dr " prefix demoted
            titled = n == n.title()
            no_pfx = not re.match(r"^(Dr|DR)\s", n)
            return (tc, no_pfx, titled, freq, len(n))

        return max(members, key=score)

    results = []
    for root, members in final.items():
        if len(members) < 2:
            continue
        canon = best_canonical(members)
        variants = [m for m in members if m != canon]
        # Compute overall confidence
        scores = [_match_score(canon, v)[0] for v in variants]
        conf = round(sum(scores) / len(scores), 3) if scores else 1.0
        # Flag suspicious: variant shares NO token with canonical
        ct = _toks(canon)
        suspicious = any(not (_toks(v) & ct) for v in variants)
        results.append(
            {
                "canonical": canon,
                "variants": sorted(variants, key=lambda x: (-counts.get(x, 0), -len(x))),
                "confidence": conf,
                "suspicious": suspicious,
                "count": len(members),
            }
        )
    results.sort(key=lambda x: (-x["count"], -x["confidence"]))
    return results


def apply_name_normalisation(
    df: "pd.DataFrame", col: str, approved_clusters: list[dict]
) -> "pd.DataFrame":
    """
    Apply approved rename clusters to a column in a copy of df.
    
    Args:
        df: Input DataFrame
        col: Column name to normalize
        approved_clusters: List of cluster dicts from detect_name_clusters
        
    Returns:
        DataFrame with normalized column values
    """
    import pandas as pd
    
    df = df.copy()
    mapping = {}
    for c in approved_clusters:
        for v in c["variants"]:
            mapping[v] = c["canonical"]
    df[col] = df[col].map(lambda x: mapping.get(x, x))
    return df
