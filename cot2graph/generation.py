"""Generate CoT traces with vLLM and segment them into steps."""

from __future__ import annotations

import random

import numpy as np
from tqdm import tqdm

from .config import resolve, stage_file
from .data import format_question, load_benchmark
from .segmentation import build_marker_pattern, segment_cot
from .utils import read_text, set_seed, write_json


def step_uncertainty(logprobs: list[dict], token_ids: list[int]) -> dict | None:
    """Uncertainty metrics for one step from vLLM top-K log-probs."""
    chosen, entropies, gaps = [], [], []
    for pos, lp_dict in enumerate(logprobs):
        if not lp_dict:
            continue
        tok = token_ids[pos] if pos < len(token_ids) else None
        if tok in lp_dict:
            chosen.append(lp_dict[tok].logprob)
        else:
            chosen.append(min(lp_dict.values(), key=lambda x: getattr(x, "rank", 0)).logprob)

        raw = np.array([lp.logprob for lp in lp_dict.values()], dtype=np.float64)
        raw -= raw.max()
        probs = np.exp(raw) / np.exp(raw).sum()
        entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
        if len(raw) >= 2:
            top = np.sort(raw)[::-1]
            gaps.append(float(top[0] - top[1]))

    if not chosen:
        return None
    chosen, entropies = np.array(chosen), np.array(entropies)
    return {
        "entropy": {
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
            "min": float(entropies.min()),
            "max": float(entropies.max()),
            "median": float(np.median(entropies)),
        },
        "logprob": {"mean": float(chosen.mean()), "sum": float(chosen.sum())},
        "beam": {"variance": float(chosen.var()), "gap_top2": float(np.mean(gaps)) if gaps else 0.0},
        "structure": {"length": len(token_ids), "unique_tokens": len(set(token_ids))},
    }


def steps_uncertainty(text: str, steps: dict, logprobs, token_ids) -> dict:
    """Map each step to its token span (by character share) and score it."""
    n_chars, n_tokens = max(len(text), 1), len(token_ids)
    out, cursor = {}, 0
    for key, step in steps.items():
        end = cursor + len(step)
        start_tok = round(cursor / n_chars * n_tokens)
        end_tok = max(min(round(end / n_chars * n_tokens), n_tokens), start_tok + 1)
        metrics = step_uncertainty(logprobs[start_tok:end_tok], token_ids[start_tok:end_tok])
        if metrics:
            out[key] = metrics
        cursor = end
    return out


def run(cfg) -> None:
    from vllm import LLM, SamplingParams

    gen = cfg.generation
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    instructions = read_text(resolve(gen.prompt))
    pattern = build_marker_pattern(cfg.segmentation.markers)
    df = load_benchmark(cfg)

    llm = LLM(
        model=cfg.models[cfg.model],
        tensor_parallel_size=gen.tensor_parallel_size,
        gpu_memory_utilization=gen.gpu_memory_utilization,
        dtype=gen.dtype,
        trust_remote_code=True,
    )
    params = SamplingParams(
        temperature=gen.temperature, max_tokens=gen.max_new_tokens, logprobs=gen.logprobs_top_k
    )

    for run_id in range(gen.num_runs):
        questions = [format_question(row, cfg.dataset, rng) for row in df.to_dict("records")]
        traces = {}
        for start in tqdm(range(0, len(questions), gen.batch_size), desc=f"run {run_id}"):
            batch = questions[start : start + gen.batch_size]
            prompts = [f"{instructions}\n\nProblem:\n{q['problem']}" for q in batch]
            for i, output in enumerate(llm.generate(prompts, params, use_tqdm=False)):
                completion = output.outputs[0]
                text = completion.text
                steps = segment_cot(text, pattern, cfg.segmentation.min_chars)
                cot = {"raw": text, "parsed": steps}
                if completion.logprobs and completion.token_ids:
                    cot["entropy"] = steps_uncertainty(
                        text, steps, completion.logprobs, list(completion.token_ids)
                    )
                traces[str(start + i)] = {**batch[i], "cot": cot}

        out = stage_file(cfg, "generation", run_id)
        write_json(traces, out, indent=1)
        print(f"Saved {len(traces)} traces to {out}")
