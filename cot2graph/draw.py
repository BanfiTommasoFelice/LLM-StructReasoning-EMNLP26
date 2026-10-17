"""Render the CoT graph of one trace with Graphviz (as in Figure 1 and Appendix J)."""

from __future__ import annotations

import html
import textwrap

from .config import run_dir, stage_file
from .utils import PROBLEM, read_json, judge_of

EDGE_COLORS = {
    "inference": "#1F9D55",
    "background info": "#215CAF",
    "observation": "#E8A33D",
    "meta-reflection": "#00BFFF",
    "continuous": "#7F7F7F",
}


def node_label(rep: str, members: list[str], text: str, judge: dict, width: int = 50) -> str:
    wrapped = "<BR/>".join(html.escape(line) for line in textwrap.wrap(text.replace("<think>", ""), width))
    rows = [f'<TR><TD BGCOLOR="lightblue"><B>{html.escape(rep)}</B> ({html.escape(", ".join(members))})</TD></TR>',
            f"<TR><TD>{wrapped}</TD></TR>"]
    for task in ("relevance", "hallucinated"):
        label = judge.get(task, {}).get("parsed", {}).get(rep, {}).get("label")
        if label:
            rows.append(f'<TR><TD ALIGN="LEFT"><I>{task}:</I> {html.escape(label)}</TD></TR>')
    return '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">' + "".join(rows) + "</TABLE>>"


def run(cfg, qid: str, graph_key: str = "matrix_compressed") -> None:
    from graphviz import Digraph

    entry = read_json(stage_file(cfg, "graphs", 0))[qid]
    graph = entry[graph_key]
    matrix, rep = graph["matrix"], graph["representative"]
    texts = {**entry["cot"]["parsed"], PROBLEM: entry["problem"]}
    judge = judge_of(entry)

    members = {}
    for node, r in rep.items():
        members.setdefault(r, []).append(node)

    dot = Digraph(format="pdf")
    dot.attr(rankdir="TB", nodesep="0.6", ranksep="1.0")
    for node in matrix:
        dot.node(node, label=node_label(node, members.get(node, [node]), texts.get(node, ""), judge),
                 shape="box", style="filled", fillcolor="white")
    for src, targets in matrix.items():
        for tgt, edge in targets.items():
            dot.edge(src, tgt, label=f"{edge['class']} ({edge['strength']})",
                     color=EDGE_COLORS.get(edge["class"], "black"))
    for src, tgt in graph.get("missing_edges", []):
        dot.edge(src, tgt, color="red", style="dashed", label="missing")

    out = run_dir(cfg) / "figures" / f"graph_{qid}_{graph_key}"
    out.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(out), cleanup=True)
    print(f"Saved {out}.pdf")
