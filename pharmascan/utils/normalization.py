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
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    if n < 0:
        return f"-{fmt_number(-n)}"
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


# ── RAMA / Patient-ID fuzzy helpers ──────────────────────────────────────────

def _norm_rama(rama_str: str) -> str:
    """Normalise a RAMA/patient-ID string for comparison (strip non-alphanumeric)."""
    return re.sub(r"[^a-z0-9]", "", str(rama_str).lower().strip())


def _rama_similarity(a: str, b: str) -> float:
    """
    Similarity score (0-1) between two RAMA numbers.
    Returns 1.0 for exact match, 0.0 if lengths differ by > 2 chars, else
    SequenceMatcher ratio on the normalised strings.
    """
    na, nb = _norm_rama(a), _norm_rama(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if abs(len(na) - len(nb)) > 2:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def detect_fuzzy_repeat_patients(
    df,
    name_thresh: float = 0.82,
    rama_thresh: float = 0.88,
) -> list[dict]:
    """
    Detect duplicate / repeat patients using a combination of:
      • Exact RAMA-number match
      • Fuzzy RAMA-number match  (similarity ≥ rama_thresh)
      • Fuzzy patient-name match  (similarity ≥ name_thresh)

    Returns a list of dicts, each representing a group of suspected
    duplicate patients:
      {
        "canonical_id":   str,   # most-common patient_id in the group
        "canonical_name": str,   # most-common patient_name in the group
        "members":        [ {patient_id, patient_name, visits, match_type, confidence} ],
        "match_types":    set,   # e.g. {"EXACT_RAMA", "FUZZY_NAME"}
        "confidence":     float, # average pairwise confidence
        "total_visits":   int,
      }
    Only groups with ≥ 2 distinct (id, name) pairs are returned.
    """
    has_id   = "patient_id"   in df.columns
    has_name = "patient_name" in df.columns

    if not has_id and not has_name:
        return []

    # ── Build unique (patient_id, patient_name, visit_count) records ─────────
    group_cols = []
    if has_id:   group_cols.append("patient_id")
    if has_name: group_cols.append("patient_name")

    agg = df.groupby(group_cols, dropna=False).size().reset_index(name="_visits")

    # Coerce to string; replace NaN-like values with empty string
    if has_id:
        agg["patient_id"]   = agg["patient_id"].fillna("").astype(str).str.strip()
    if has_name:
        agg["patient_name"] = agg["patient_name"].fillna("").astype(str).str.strip()

    records = agg.to_dict("records")
    n = len(records)
    if n < 2:
        return []

    # ── Union-Find (path compression) ────────────────────────────────────────
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    # Store match metadata per pair for confidence/type tracking
    pair_meta: dict[tuple[int, int], tuple[float, str]] = {}

    # Cap at 800 records to keep O(n²) feasible; prioritise by visit count
    if n > 800:
        records = sorted(records, key=lambda r: -r.get("_visits", 0))[:800]
        n = 800

    # ── Pairwise comparison ───────────────────────────────────────────────────
    for i in range(n):
        ri = records[i]
        id_i   = ri.get("patient_id",   "") if has_id   else ""
        nam_i  = ri.get("patient_name", "") if has_name else ""
        norm_i = _norm_rama(id_i)

        for j in range(i + 1, n):
            rj = records[j]
            id_j   = rj.get("patient_id",   "") if has_id   else ""
            nam_j  = rj.get("patient_name", "") if has_name else ""
            norm_j = _norm_rama(id_j)

            match_type = None
            confidence = 0.0

            # 1) Exact RAMA match (non-empty)
            if has_id and norm_i and norm_j and norm_i == norm_j:
                match_type = "EXACT_RAMA"
                confidence = 1.0

            # 2) Fuzzy RAMA match
            elif has_id and norm_i and norm_j:
                rs = _rama_similarity(id_i, id_j)
                if rs >= rama_thresh:
                    match_type = "FUZZY_RAMA"
                    confidence = rs

            # 3) Fuzzy name match
            if has_name and nam_i and nam_j:
                ns, _why = _match_score(nam_i, nam_j)
                if ns >= name_thresh:
                    if match_type is None:
                        match_type = "FUZZY_NAME"
                        confidence = ns
                    else:
                        # Both RAMA and name match → strongest signal
                        match_type = "MULTI_MATCH"
                        confidence = round((confidence + ns) / 2, 4)

            if match_type:
                union(i, j)
                key = (min(i, j), max(i, j))
                # Keep highest-confidence label for this pair
                if key not in pair_meta or confidence > pair_meta[key][0]:
                    pair_meta[key] = (confidence, match_type)

    # ── Build result groups ───────────────────────────────────────────────────
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    results = []
    for root, members_idx in clusters.items():
        if len(members_idx) < 2:
            continue

        # Collect pairwise metadata within this cluster
        cluster_types: set = set()
        cluster_confs: list = []
        for ii in range(len(members_idx)):
            for jj in range(ii + 1, len(members_idx)):
                key = (min(members_idx[ii], members_idx[jj]),
                       max(members_idx[ii], members_idx[jj]))
                if key in pair_meta:
                    conf, mtype = pair_meta[key]
                    cluster_types.add(mtype)
                    cluster_confs.append(conf)

        avg_conf = round(sum(cluster_confs) / len(cluster_confs), 3) if cluster_confs else 0.0

        # Canonical ID/name = most frequent member
        member_records = [records[i] for i in members_idx]
        canonical_id   = max(
            (r.get("patient_id",   "") for r in member_records),
            key=lambda x: sum(r.get("patient_id","") == x for r in member_records),
        ) if has_id else ""
        canonical_name = max(
            (r.get("patient_name", "") for r in member_records),
            key=lambda x: sum(r.get("patient_name","") == x for r in member_records),
        ) if has_name else ""

        total_visits = sum(r.get("_visits", 0) for r in member_records)

        members_out = []
        for r in member_records:
            # find best match_type for this member vs canonical
            best_type = "UNKNOWN"
            best_conf = 0.0
            idx_r = records.index(r) if r in records else -1
            for key, (c, t) in pair_meta.items():
                if idx_r in key:
                    if c > best_conf:
                        best_conf = c
                        best_type = t
            members_out.append({
                "patient_id":   r.get("patient_id",   ""),
                "patient_name": r.get("patient_name", ""),
                "visits":       int(r.get("_visits", 0)),
                "match_type":   best_type,
                "confidence":   round(best_conf, 3),
            })

        results.append({
            "canonical_id":   canonical_id,
            "canonical_name": canonical_name,
            "members":        members_out,
            "match_types":    cluster_types,
            "confidence":     avg_conf,
            "total_visits":   int(total_visits),
        })

    results.sort(key=lambda x: (-x["confidence"], -x["total_visits"]))
    return results
