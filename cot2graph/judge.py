"""Step- and trace-level labels from an LLM judge (Section 4.1)."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import resolve, stage_file
from .utils import append_jsonl, cot_steps, fill, read_json, read_jsonl, read_text

PROMPT_FILES = {
    "final_answer": "prompts/judge_final_answer.txt",
    "hallucinated": "prompts/judge_hallucinated.txt",
    "relevance": "prompts/judge_relevance.txt",
    "trace_summary": "prompts/judge_trace_summary.txt",
    "node_assignment": "prompts/judge_node_assignment.txt",
}
_STEP_LABEL = re.compile(r"\[(?:step\s*)?(\d+)\]\s*(\w+)\s*\|\s*(?:WHY:\s*)?(.*)")
_SUMMARY = re.compile(r"([A-Z])\)\s*(.*)")


def parse_final_answer(raw: str) -> dict:
    status, steps = "Unknown", []
    for line in raw.splitlines():
        line = line.strip()
        low = line.lower()
        if low.startswith("final answer correct:"):
            value = line.split(":", 1)[1].strip().lower()
            for label in ("correct", "partial", "incorrect"):
                if value.startswith(label):
                    status = label.capitalize()
                    break
        elif low.startswith("final answer in cot:"):
            steps = [int(x) for x in re.findall(r"\d+", line)]
    return {"final_answer_correct": status, "final_answer_in_cot": steps}


def parse_step_labels(raw: str) -> dict:
    parsed = {}
    for line in raw.splitlines():
        m = _STEP_LABEL.match(line.strip())
        if m:
            parsed[m.group(1)] = {"label": m.group(2), "why": m.group(3)}
    return parsed


def parse_summary(raw: str) -> dict:
    parsed = {}
    for line in raw.splitlines():
        m = _SUMMARY.match(line.strip())
        if m:
            parsed[m.group(1)] = m.group(2)
    return parsed


def covers_all_steps(parsed: dict, n_steps: int) -> bool:
    try:
        return sorted(int(k) for k in parsed) == list(range(n_steps))
    except ValueError:
        return False


PARSERS = {
    "final_answer": (parse_final_answer, lambda p, n: True),
    "hallucinated": (parse_step_labels, covers_all_steps),
    "relevance": (parse_step_labels, covers_all_steps),
    "trace_summary": (parse_summary, lambda p, n: len(p) > 0),
    "node_assignment": (parse_step_labels, covers_all_steps),
}


class Judge:
    def __init__(self, cfg):
        from openai import OpenAI

        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("Set OPENAI_API_KEY to run the judge.")
        self.cfg = cfg.judge
        self.client = OpenAI()
        self.templates = {task: read_text(resolve(path)) for task, path in PROMPT_FILES.items()}

    def ask(self, prompt: str, max_retries: int = 100, wait: float = 15.0, max_wait: float = 120.0) -> str:
        from openai import APIError, APITimeoutError, RateLimitError

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=self.cfg.max_output_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except (RateLimitError, APIError, APITimeoutError) as err:
                if attempt == max_retries - 1:
                    raise
                tqdm.write(f"API error ({err}); retrying in {wait:.0f}s")
                time.sleep(wait)
                wait = min(wait * 2, max_wait)
        return ""

    def label(self, entry: dict) -> dict:
        steps = cot_steps(entry)
        n_steps = len(steps)
        fields = {
            "problem": entry["problem"],
            "solution": entry.get("solution"),
            "answer": entry.get("answer"),
            "cot": "\n".join(f"[step {k}]: {steps[k]}" for k in sorted(steps, key=int)),
            "last_step": n_steps - 1,
        }
        labels = dict(entry.get("judge", {}))
        for task in self.cfg.tasks:
            if task in labels:
                continue
            if task == "node_assignment":
                summary = labels.get("trace_summary", {}).get("parsed")
                if not summary:
                    continue
                fields["summary"] = "\n".join(f"{k}) {v}" for k, v in summary.items())
            prompt = fill(self.templates[task], **fields)

            parse, is_complete = PARSERS[task]
            for attempt in range(1, self.cfg.max_attempts + 1):
                raw = self.ask(prompt)
                parsed = parse(raw)
                if is_complete(parsed, n_steps):
                    break
            labels[task] = {"raw": raw, "parsed": parsed, "attempts": attempt}
        return labels


def run(cfg) -> None:
    judge = Judge(cfg)
    for run_id in range(cfg.generation.num_runs):
        traces = read_json(stage_file(cfg, "generation", run_id))
        out = stage_file(cfg, "judge", run_id, "jsonl")
        done = read_jsonl(out)
        todo = {k: v for k, v in traces.items() if k not in done}
        print(f"run {run_id}: {len(todo)} traces to label ({len(done)} already done)")

        with ThreadPoolExecutor(max_workers=cfg.judge.num_threads) as pool:
            futures = {pool.submit(judge.label, entry): qid for qid, entry in todo.items()}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"run {run_id}"):
                qid = futures[future]
                try:
                    append_jsonl(out, qid, {**todo[qid], "judge": future.result()})
                except Exception as err:
                    tqdm.write(f"[error] id={qid}: {err}")
