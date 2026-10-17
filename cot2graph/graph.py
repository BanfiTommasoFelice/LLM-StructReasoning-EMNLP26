"""Merge edges and judge labels, then build the compressed CoT graph (Section 3.5)."""

from __future__ import annotations

from .config import stage_file
from .utils import (
    PROBLEM,
    SOLUTION,
    cot_steps,
    edge_class,
    judge_of,
    node_key,
    read_jsonl,
    sorted_nodes,
    write_json,
)

DEFAULT_LABELS = {"hallucinated": "partial", "relevance": "relevant"}
VALID_LABELS = {
    "hallucinated": {"correct", "partial", "hallucinated"},
    "relevance": {"relevant", "redundant", "irrelevant"},
}


def prune_matrix(matrix: dict, min_strength: dict) -> dict:
    """Drop the answer node, ``none`` edges and edges below the per-type threshold."""
    pruned = {}
    for src, targets in matrix.items():
        if src == SOLUTION:
            continue
        kept = {}
        for tgt, edge in targets.items():
            cls = edge_class(edge)
            if tgt == SOLUTION or cls not in min_strength:
                continue
            if edge.get("strength", 0) >= min_strength[cls]:
                kept[tgt] = {"class": cls, "strength": edge.get("strength", 0)}
        pruned[src] = kept
    return pruned


def clean_labels(judge: dict, n_steps: int) -> dict:
    """Give every step a valid label so downstream code never meets a gap."""
    fa = judge.setdefault("final_answer", {}).setdefault("parsed", {})
    if fa.get("final_answer_correct") not in {"Correct", "Partial", "Incorrect"}:
        fa["final_answer_correct"] = "Incorrect"
    fa.setdefault("final_answer_in_cot", [])

    for task, default in DEFAULT_LABELS.items():
        if task not in judge:
            continue
        parsed = judge[task].setdefault("parsed", {})
        for k in map(str, range(n_steps)):
            if parsed.get(k, {}).get("label") not in VALID_LABELS[task]:
                parsed[k] = {**parsed.get(k, {}), "label": default}
    return judge


class UnionFind:
    """Union-Find whose representative is always the earliest step."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = sorted_nodes([self.find(a), self.find(b)])
        if ra != rb:
            self.parent[rb] = ra


def compress_graph(matrix_plain: dict, tie_break: list[str], stats: dict | None = None) -> dict:
    """Merge repetition components and resolve edges that collapse onto one pair."""
    matrix = matrix_plain["matrix"]
    nodes = list(matrix)
    rank = {cls: i for i, cls in enumerate(tie_break)}
    uf = UnionFind(nodes)
    for src, targets in matrix.items():
        for tgt, edge in targets.items():
            if edge_class(edge) == "repetition":
                uf.union(src, tgt)
    rep = {n: uf.find(n) for n in nodes}

    compressed = {r: {} for r in sorted_nodes(set(rep.values()))}
    for src in nodes:
        for tgt, edge in matrix[src].items():
            cls = edge_class(edge)
            rs, rt = rep[src], rep[tgt]
            if cls in ("none", "repetition") or rs == rt:
                continue
            edge = {"class": cls, "strength": edge.get("strength", 0)}
            old = compressed[rs].get(rt)
            if old is None:
                compressed[rs][rt] = edge
                continue
            if stats is not None:
                stats["conflicts"] = stats.get("conflicts", 0) + 1
                if old["strength"] == edge["strength"]:
                    stats["ties"] = stats.get("ties", 0) + 1
                    stats["same_type_ties"] = stats.get("same_type_ties", 0) + (old["class"] == cls)
            if edge["strength"] > old["strength"] or (
                edge["strength"] == old["strength"]
                and rank.get(cls, len(rank)) < rank.get(old["class"], len(rank))
            ):
                compressed[rs][rt] = edge

    plain_rep = matrix_plain.get("representative", {})
    full_rep = {n: rep.get(plain_rep.get(n, n), plain_rep.get(n, n)) for n in plain_rep if n != SOLUTION}
    full_rep.update(rep)
    compressed = {r: dict(sorted(t.items(), key=lambda kv: node_key(kv[0]))) for r, t in compressed.items()}
    return {"matrix": compressed, "representative": full_rep, "missing_edges": missing_edges(compressed, full_rep)}


def missing_edges(matrix: dict, rep: dict) -> list[list[str]]:
    """Consecutive CoT steps with no dependency between their representatives."""
    path = [PROBLEM] + sorted((n for n in matrix if n.isdigit()), key=int)
    missing = set()
    for a, b in zip(path, path[1:]):
        ra, rb = rep.get(a, a), rep.get(b, b)
        if ra != rb and rb not in matrix.get(ra, {}):
            missing.add((ra, rb))
    return [list(e) for e in sorted(missing)]


def build_tree(entry: dict, graph: dict) -> dict:
    """LCoT2Tree baseline (Jiang et al., 2025): a tree over the judge's thought groups."""
    assignment = judge_of(entry)["node_assignment"]["parsed"]
    matrix, rep = graph["matrix"], graph["representative"]

    def group(node):
        value = assignment.get(node)
        return value.get("label") if isinstance(value, dict) else value

    steps = sorted(cot_steps(entry), key=int)
    letters = sorted({group(n) for n in steps if group(n) is not None})
    level = {letter: i + 1 for i, letter in enumerate(letters)}

    tree = {PROBLEM: {}}
    last_at_level = {0: PROBLEM}
    for node in steps:
        lvl = level.get(group(node), 1)
        tree[node] = {}
        latest = max(last_at_level)
        parent = last_at_level[latest] if lvl > latest else last_at_level.get(lvl - 1, PROBLEM)

        rp, rn = rep.get(parent, parent), rep.get(node, node)
        if rp == rn:
            edge = {"class": "repetition", "strength": 10}
        else:
            edge = dict(matrix.get(rp, {}).get(rn, {"class": "continuous", "strength": 10}))
            if edge.get("class") == "none":
                edge["class"] = "continuous"
        tree[parent][node] = edge
        last_at_level[lvl] = node
    return {"matrix": tree, "representative": {n: n for n in tree}}


def build_entry(entry: dict, cfg, stats: dict | None = None) -> dict:
    entry = dict(entry)
    entry["judge"] = clean_labels(judge_of(entry), len(cot_steps(entry)))
    entry.pop("judge_advanced", None)
    plain = entry["matrix_plain"]
    plain = {**plain, "matrix": prune_matrix(plain["matrix"], cfg.edges.min_strength)}
    entry["matrix_plain"] = plain
    entry["matrix_compressed"] = compress_graph(plain, cfg.edges.tie_break, stats)
    if "node_assignment" in entry["judge"]:
        entry["tree_compressed"] = build_tree(entry, entry["matrix_compressed"])
    return entry


def run(cfg) -> None:
    for run_id in range(cfg.generation.num_runs):
        edges = read_jsonl(stage_file(cfg, "edges", run_id, "jsonl"))
        labels = read_jsonl(stage_file(cfg, "judge", run_id, "jsonl"))
        missing = set(edges) ^ set(labels)
        if missing:
            raise ValueError(f"run {run_id}: edges and judge outputs disagree on ids {sorted(missing)[:10]}")

        stats = {}
        graphs = {}
        for qid in sorted(labels, key=int):
            entry = {**labels[qid], "matrix_plain": edges[qid]["matrix_plain"]}
            graphs[qid] = build_entry(entry, cfg, stats)

        out = stage_file(cfg, "graphs", run_id)
        write_json(graphs, out, indent=1)
        n_nodes = sum(len(g["matrix_plain"]["matrix"]) for g in graphs.values())
        n_kept = sum(len(g["matrix_compressed"]["matrix"]) for g in graphs.values())
        ties, same = stats.get("ties", 0), stats.get("same_type_ties", 0)
        print(f"run {run_id}: {len(graphs)} graphs -> {out}")
        print(f"  nodes merged away: {1 - n_kept / max(n_nodes, 1):.1%} | "
              f"equal-strength conflicts with the same type: {same / max(ties, 1):.1%}")
