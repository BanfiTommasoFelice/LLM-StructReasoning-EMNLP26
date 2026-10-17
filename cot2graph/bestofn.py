"""Best-of-N selection with structural, reward-model and uncertainty verifiers (Section 5.5)."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from .config import run_dir, stage_file
from .features import to_pyg
from .train_graph import DEVICE, train_model
from .utils import cot_steps, is_correct, read_json, set_seed, write_json

DIRECTION = {"entropy": "min", "prm": "max", "orm": "max", "lr_length": "max", "gnn": "max"}
DISPLAY = {"entropy": "Entropy", "prm": "PRM", "orm": "ORM", "lr_length": "Length LR", "gnn": "CoT2Graph"}


def display(name: str) -> str:
    return " + ".join(DISPLAY.get(part, part) for part in name.split(" + "))


def mean_entropy(entry: dict) -> float:
    """Mean token entropy over the steps of a trace (lower = more confident)."""
    values = [v["entropy"]["mean"] for v in entry.get("cot", {}).get("entropy", {}).values()]
    return float(np.mean(values)) if values else float("inf")


def fuse(score_lists: list, directions: list) -> np.ndarray:
    """Sum of per-question min-max normalised scores (higher = better)."""
    total = np.zeros(len(score_lists[0]))
    for scores, direction in zip(score_lists, directions):
        s = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        span = s.max() - s.min()
        s = (s - s.min()) / span if span > 0 else np.zeros_like(s)
        total += 1.0 - s if direction == "min" else s
    return total


class RewardModels:
    """Outcome (Skywork) and process (Math-Shepherd) reward models."""

    def __init__(self, cfg):
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        b = cfg.bestofn
        self.max_length = b.max_length
        self.orm_tok = AutoTokenizer.from_pretrained(b.orm)
        self.orm = AutoModelForSequenceClassification.from_pretrained(
            b.orm, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.prm_tok = AutoTokenizer.from_pretrained(b.prm)
        self.prm = AutoModelForCausalLM.from_pretrained(
            b.prm, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.good = self.prm_tok.encode("+", add_special_tokens=False)[0]
        self.step_tag = self.prm_tok.encode("ки", add_special_tokens=False)[-1]

    @torch.no_grad()
    def orm_score(self, entry: dict) -> float:
        text = f"{entry.get('problem', '')}\n\n{entry['cot']['raw']}"
        inputs = self.orm_tok(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        return self.orm(**inputs.to(self.orm.device)).logits[0, 0].item()

    @torch.no_grad()
    def prm_score(self, entry: dict) -> float:
        steps = cot_steps(entry)
        text = entry.get("problem", "") + "\n" + " ки\n".join(steps[k] for k in sorted(steps, key=int)) + " ки"
        inputs = self.prm_tok(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        logits = self.prm(**inputs.to(self.prm.device)).logits[0]
        ids = inputs["input_ids"][0].tolist()
        scores = [torch.softmax(logits[i - 1], -1)[self.good].item() for i, t in enumerate(ids) if t == self.step_tag and i > 0]
        return float(np.mean(scores)) if scores else 0.0


def split_questions(qids: list[str], cfg):
    b = cfg.bestofn
    rng = random.Random(b.seed)
    qids = qids[:]
    rng.shuffle(qids)
    n_train = int(len(qids) * b.train_ratio)
    train, test = qids[:n_train], qids[n_train:]
    n_val = int(len(train) * b.val_ratio)
    return train[n_val:], train[:n_val], test


def run(cfg) -> None:
    b = cfg.bestofn
    set_seed(b.seed, deterministic=True)
    runs = [read_json(stage_file(cfg, "graphs", r)) for r in range(b.num_runs)]
    qids = sorted(set.intersection(*(set(r) for r in runs)), key=int)
    train_q, val_q, test_q = split_questions(qids, cfg)
    print(f"{len(qids)} questions x {b.num_runs} runs | train {len(train_q)} / val {len(val_q)} / test {len(test_q)}")

    def graphs_of(questions):
        return [to_pyg(runs[r][q], "matrix_compressed", cfg.features, uid=f"{q}_{r}")[0]
                for q in questions for r in range(b.num_runs)]

    gnn = train_model(cfg, graphs_of(train_q), graphs_of(val_q), run_dir(cfg) / "checkpoints" / "bestofn.pt")
    gnn.eval()
    lengths = [[len(cot_steps(runs[r][q]))] for q in train_q for r in range(b.num_runs)]
    labels = [is_correct(runs[r][q]) for q in train_q for r in range(b.num_runs)]
    length_lr = LogisticRegression(max_iter=3000).fit(lengths, labels)
    rewards = RewardModels(cfg)

    @torch.no_grad()
    def gnn_score(entry):
        g = to_pyg(entry, "matrix_compressed", cfg.features)[0].to(DEVICE)
        batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        return F.softmax(gnn(g.x, g.edge_index, g.edge_attr, batch), -1)[0, 1].item()

    scorers = {
        "entropy": mean_entropy,
        "prm": rewards.prm_score,
        "orm": rewards.orm_score,
        "lr_length": lambda e: float(length_lr.predict_proba([[len(cot_steps(e))]])[0, 1]),
        "gnn": gnn_score,
    }

    results = {}
    for q in test_q:
        candidates = [runs[r][q] for r in range(b.num_runs)]
        truth = [is_correct(c) for c in candidates]
        row = {"labels": truth, "scores": {}, "correct": {"mean": float(np.mean(truth)), "oracle": max(truth)}}
        for name, scorer in scorers.items():
            scores = [scorer(c) for c in candidates]
            pick = int(np.argmin(scores) if DIRECTION[name] == "min" else np.argmax(scores))
            row["scores"][name] = scores
            row["correct"][name] = truth[pick]
        for combo in b.fusions:
            name = " + ".join(combo)
            fused = fuse([row["scores"][m] for m in combo], [DIRECTION[m] for m in combo])
            row["correct"][name] = truth[int(np.argmax(fused))]
        results[q] = row

    accuracy = pd.DataFrame([r["correct"] for r in results.values()]).mean() * 100
    print("\nBest-of-N accuracy (%):")
    print(accuracy.rename(display).round(1).to_string())

    flat = {m: np.concatenate([r["scores"][m] for r in results.values()]) for m in scorers}
    signed = {m: (-v if DIRECTION[m] == "min" else v) for m, v in flat.items()}
    corr = pd.DataFrame({a: {b_: spearmanr(signed[a], signed[b_])[0] for b_ in signed} for a in signed})
    print("\nSpearman correlation of selection scores:")
    print(corr.rename(index=display, columns=display).round(2).to_string())

    out = run_dir(cfg) / "bestofn"
    write_json(results, out / "results.json")
    write_json({"accuracy": accuracy.to_dict(), "spearman": corr.to_dict()}, out / "summary.json")
    print(f"Saved results to {out}")
