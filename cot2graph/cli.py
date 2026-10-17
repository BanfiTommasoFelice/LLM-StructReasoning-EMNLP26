"""Command-line interface: ``python -m cot2graph <stage> [key=value ...]``."""

from __future__ import annotations

import argparse
import importlib

STAGES = {
    "generate": ("generation", "Generate CoT traces with vLLM and segment them into steps"),
    "annotate-edges": ("edges", "Label typed, weighted dependencies between step pairs"),
    "judge": ("judge", "Label final-answer correctness, relevance and hallucination"),
    "build-graphs": ("graph", "Merge annotations and build the compressed CoT graphs"),
    "train-graph": ("train_graph", "Predict final-answer correctness from graph structure"),
    "train-node": ("train_node", "Predict step relevance and hallucination from structure"),
    "explain": ("explainer_analysis", "Analyse GNNExplainer node importances"),
    "efficiency": ("efficiency", "Decompose redundancy into verification and path inefficiency"),
    "bestofn": ("bestofn", "Best-of-N selection with structural and baseline verifiers"),
    "draw": ("draw", "Render the CoT graph of one trace"),
}


def main(argv=None) -> None:
    from .config import load_config

    parser = argparse.ArgumentParser(prog="cot2graph", description="CoT2Graph pipeline")
    sub = parser.add_subparsers(dest="stage", required=True, metavar="stage")
    for name, (_, help_text) in STAGES.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("-c", "--config", default=None, help="YAML config (default: configs/default.yaml)")
        p.add_argument("overrides", nargs="*", help="config overrides, e.g. dataset=MMLU-Pro gnn.lr=5e-4")
        if name == "draw":
            p.add_argument("--id", required=True, help="question id to draw")
            p.add_argument("--graph", default="matrix_compressed", help="matrix_compressed or tree_compressed")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    module = importlib.import_module(f"cot2graph.{STAGES[args.stage][0]}")
    if args.stage == "draw":
        module.run(cfg, args.id, args.graph)
    else:
        module.run(cfg)


if __name__ == "__main__":
    main()
