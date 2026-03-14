#!/usr/bin/env python3
# Prepare RL data (verl-compatible parquet/jsonl) for FiberPO/GRPO.

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare RL data for FiberPO/GRPO (verl-compatible).")
    p.add_argument("--output", type=str, default="data/train.parquet")
    p.add_argument("--format", type=str, choices=["parquet", "jsonl"], default="parquet")
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--prompt_key", type=str, default="prompt")
    p.add_argument("--reward_fn_key", type=str, default="data_source")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    templates = [
        "Question: {q}\nLet's think step by step.",
        "Solve the following math problem.\n{q}",
    ]
    problems = [
        "Tom has 3 apples. He buys 2 more. How many apples does he have?",
        "A store has 10 toys. They sell 4. How many are left?",
        "There are 5 birds on a tree. 2 fly away. How many remain?",
    ]

    rows = []
    for i in range(args.num):
        q = problems[i % len(problems)]
        prompt = templates[i % len(templates)].format(q=q)
        rows.append({args.prompt_key: prompt, args.reward_fn_key: "math"})

    if args.format == "jsonl":
        with open(args.output, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} lines to {args.output}")
    else:
        try:
            import pandas as pd
            pd.DataFrame(rows).to_parquet(args.output, index=False)
            print(f"Wrote {len(rows)} rows to {args.output}")
        except ImportError:
            out = args.output.replace(".parquet", ".jsonl")
            with open(out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Wrote {len(rows)} lines to {out} (install pandas+pyarrow for parquet)")


if __name__ == "__main__":
    main()
