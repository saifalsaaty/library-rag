"""Figure 1 — system and evaluation architecture.

    python scripts/make_architecture_figure.py

Needs the Graphviz binaries (`dot` on the PATH). Produces
results/figures/fig1_architecture.{png,pdf}. The figure describes the study as run:
it makes no performance claims (numbers belong in the Results section).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "figures"

DATA = 'fillcolor="#e8f1fb", color="#2a78d6"'
PROC = 'fillcolor="#fdf0e9", color="#eb6834"'
GEN = 'fillcolor="#e7f6f0", color="#1baf7a"'
EVAL = 'fillcolor="#f4f4f2", color="#52514e"'

DOT = f"""
digraph fig1 {{
  rankdir=TB; dpi=300; nodesep=0.35; ranksep=0.5; fontname="Arial"; newrank=true;
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, margin="0.14,0.07"];
  edge [fontname="Arial", fontsize=8, color="#52514e", arrowsize=0.6];

  subgraph cluster_data {{
    label="A. Bibliographic data"; style=dashed; color="#9a998f"; fontsize=10; labeljust=l;
    marc  [label="MARC21 / CSV export\n(supported; not used here)", {DATA}, style="rounded,filled,dashed"];
    oai   [label="OAI-PMH harvest (Dublin Core)\n__SOURCE__", {DATA}];
    clean [label="One record schema, cleaning,\nde-duplication, Arabic normalization", {PROC}];
    tmpl  [label="Record text: field templates\n(T, T+S, T+A, T+S+A, full);\nmetadata in AR+EN / EN / AR", {PROC}];
    {{ rank=same; marc; oai; clean; tmpl; }}
    marc -> oai [style=invis]; oai -> clean; clean -> tmpl;
  }}

  subgraph cluster_ret {{
    label="B. Retrieval"; style=dashed; color="#9a998f"; fontsize=10; labeljust=l;
    query [label="Request\n(Arabic / English)", {GEN}];
    retr  [label="Lexical: BM25 (Arabic light10 + English stemming)\nDense (GPU): BGE-M3, mE5, MiniLM; exact cosine search", {PROC}];
    rrf   [label="Hybrid fusion\n(RRF)", {PROC}];
    {{ rank=same; query; retr; rrf; }}
    query -> retr [label="search"]; retr -> rrf;
  }}

  subgraph cluster_gen {{
    label="C. Grounded generation (local)"; style=dashed; color="#9a998f"; fontsize=10; labeljust=r;
    answer [label="Recommendations with\nverified citations,\nor abstention", {GEN}];
    guard  [label="Verification guard\n(answer rebuilt from\nverified records)", {GEN}];
    verify [label="Citation verification\n(catalog matching)", {GEN}];
    llm    [label="Small local LLM\n(Ollama, 1B-7B, Q4)", {GEN}];
    prompt [label="Request + top-k numbered\nrecords + grounding prompt", {GEN}];
    {{ rank=same; answer; guard; verify; llm; prompt; }}
    answer -> guard -> verify -> llm -> prompt [dir=back];     // drawn right to left
  }}

  subgraph cluster_eval {{
    label="D. Evaluation"; style=dashed; color="#9a998f"; fontsize=10; labeljust=l;
    metrics [label="Retrieval: nDCG, MRR, recall  |  Generation: ungrounded citations,\nabstention, latency  |  Human validation of the citation detector", {EVAL}];
    judge   [label="Librarian relevance judgments\n(pooled, two annotators)", {EVAL}];
    {{ rank=same; metrics; judge; }}
    metrics -> judge [dir=back];
  }}

  marc -> query -> answer -> metrics [style=invis, weight=50];   // left-align the four rows
  marc -> clean [style=dashed, constraint=false];
  tmpl -> retr [label=" indexing"];
  rrf -> prompt [label=" top-k records"];
  answer -> metrics [style=dotted, arrowhead=none];
  rrf -> judge [style=dotted, arrowhead=none, constraint=false];
}}
"""


def main() -> None:
    dot = shutil.which("dot")
    if not dot:
        raise SystemExit("Graphviz 'dot' was not found on the PATH")
    OUT.mkdir(parents=True, exist_ok=True)
    # the _anon version hides the institution for double-anonymous review
    versions = {"fig1_architecture": "21 University of Mosul OJS journals",
                "fig1_architecture_anon": "21 OJS journals of one university"}
    for name, source in versions.items():
        for fmt in ("png", "pdf"):
            subprocess.run([dot, f"-T{fmt}", "-o", str(OUT / f"{name}.{fmt}")],
                           input=DOT.replace("__SOURCE__", source).encode("utf-8"), check=True)
        print(f"Figure 1 → {OUT / (name + '.png')}")


if __name__ == "__main__":
    main()
