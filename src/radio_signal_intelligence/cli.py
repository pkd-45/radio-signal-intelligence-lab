from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_dataset
from .evaluate import evaluate_model
from .llm_brief import build_candidate_brief
from .train import train_model


def run_demo(config_path: str | Path) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    npz = build_dataset(
        out,
        n_per_class=int(config["n_per_class"]),
        image_size=int(config["image_size"]),
        seed=int(config["seed"]),
    )
    checkpoint = train_model(
        npz,
        out,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        seed=int(config["seed"]),
    )
    metrics = evaluate_model(npz, checkpoint, out)
    print(json.dumps(metrics, indent=2))
    print(f"summary plot: {out / 'radio_signal_intelligence_summary.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Radio Signal Intelligence Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="build data, train, evaluate, and plot")
    demo.add_argument("--config", default="configs/demo.json")

    make_data = sub.add_parser("make-data", help="create deterministic synthetic ON/OFF data")
    make_data.add_argument("--outdir", default="results/local")
    make_data.add_argument("--n-per-class", type=int, default=80)
    make_data.add_argument("--image-size", type=int, default=64)
    make_data.add_argument("--seed", type=int, default=20260720)

    brief = sub.add_parser("brief", help="write an optional bounded LLM review brief")
    brief.add_argument("ranking_csv")
    brief.add_argument("--output", default="candidate_review_brief.md")
    brief.add_argument("--top-n", type=int, default=10)

    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.config)
    elif args.command == "make-data":
        path = build_dataset(args.outdir, args.n_per_class, args.image_size, args.seed)
        print(path)
    elif args.command == "brief":
        print(build_candidate_brief(args.ranking_csv, args.output, args.top_n))


if __name__ == "__main__":
    main()
