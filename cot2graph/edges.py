"""Annotate typed, weighted dependencies between step pairs (Section 3.4)."""

from __future__ import annotations

import json
import re
import time

from .config import resolve, stage_file
from .utils import PROBLEM, SOLUTION, append_jsonl, cot_steps, fill, read_json, read_jsonl, read_text, set_seed

EDGE_CLASSES = {"background info", "observation", "inference", "meta-reflection", "repetition", "none"}
_JSON_RE = re.compile(r'(\{\s*"class"\s*:\s*"(.*?)"\s*,\s*"strength"\s*:\s*(\d+)\s*\})')


def parse_edge(raw: str) -> dict:
    """Parse the annotator output into ``{"class", "strength"}``."""
    match = _JSON_RE.search(raw.strip())
    if not match:
        return {"class": "not_found", "strength": 0}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"class": "failed_parsing", "strength": 0}
    cls, strength = data.get("class", "none"), int(data.get("strength", 0))
    if cls not in EDGE_CLASSES or strength < 0:
        return {"class": "none", "strength": 0}
    return {"class": cls, "strength": min(strength, 10)}


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] if ca == cb else 1 + min(prev[j], cur[j - 1], prev[j - 1]))
        prev = cur
    return prev[-1]


def representatives(texts: dict[str, str], threshold: int) -> dict[str, str]:
    """Map near-identical steps (edit distance < threshold) to their first occurrence."""
    nodes = list(texts)
    rep = {n: n for n in nodes}
    for i, a in enumerate(nodes[:-1]):
        for b in nodes[i + 1 :]:
            if texts[a] == texts[b] or edit_distance(texts[a], texts[b]) < threshold:
                rep[b] = rep[a]
    return rep


def build_pairs(entry: dict, template: str, threshold: int):
    """Prompts for every ordered pair (i < j) of representative nodes."""
    steps = cot_steps(entry)
    order = [PROBLEM] + sorted(steps, key=int) + [SOLUTION]
    texts = {PROBLEM: entry.get("problem", ""), SOLUTION: entry.get("answer", ""), **steps}
    texts = {n: str(texts[n]) for n in order}
    rep = representatives(texts, threshold)

    pairs, prompts = [], []
    for i, src in enumerate(order[:-1]):
        for j in range(i + 1, len(order)):
            tgt = order[j]
            if rep[src] != src or rep[tgt] != tgt:
                continue
            pairs.append((src, tgt))
            prompts.append(
                fill(template, problem=texts[PROBLEM], src_idx=i, src_text=texts[src], tgt_idx=j, tgt_text=texts[tgt])
            )
    nodes = [n for n in order if rep[n] == n]
    return nodes, pairs, prompts, rep


def run(cfg) -> None:
    from vllm import LLM, SamplingParams

    ec = cfg.edges
    set_seed(cfg.seed)
    template = read_text(resolve(ec.prompt))
    llm = LLM(
        model=ec.model,
        tensor_parallel_size=ec.tensor_parallel_size,
        gpu_memory_utilization=ec.gpu_memory_utilization,
        max_model_len=ec.max_model_len,
        dtype=ec.dtype,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=ec.temperature, max_tokens=ec.max_new_tokens)

    for run_id in range(cfg.generation.num_runs):
        traces = read_json(stage_file(cfg, "generation", run_id))
        out = stage_file(cfg, "edges", run_id, "jsonl")
        done = set(read_jsonl(out))
        todo = [k for k in traces if k not in done and cot_steps(traces[k])]
        print(f"run {run_id}: {len(todo)} traces to annotate ({len(done)} already done)")

        for n, qid in enumerate(todo, 1):
            entry, start = traces[qid], time.time()
            nodes, pairs, prompts, rep = build_pairs(entry, template, ec.dedup_edit_distance)
            results = []
            for i in range(0, len(prompts), ec.batch_size):
                outputs = llm.generate(prompts[i : i + ec.batch_size], params, use_tqdm=False)
                results.extend(parse_edge(o.outputs[0].text) for o in outputs)

            matrix = {node: {} for node in nodes}
            for (src, tgt), res in zip(pairs, results):
                if res["class"] in EDGE_CLASSES:
                    matrix[src][tgt] = res
            entry["matrix_plain"] = {"matrix": matrix, "representative": rep}
            append_jsonl(out, qid, entry)
            print(f"[{n}/{len(todo)}] id={qid} nodes={len(nodes)} pairs={len(pairs)} "
                  f"time={(time.time() - start) / 60:.2f} min")
