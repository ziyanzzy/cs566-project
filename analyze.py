import json
from pathlib import Path
from collections import defaultdict


def analyze_logs(logs_dir="./logs"):
    logs_dir = Path(logs_dir)

    with open(logs_dir / "summary.json") as f:
        summary = json.load(f)

    ranks = summary["rank"]
    final_scores = summary["kyoku"][-1]["end_kyoku_scores"]
    kyoku_count = summary["kyoku_count"]

    print("=" * 40)
    print("Game Result")
    print("=" * 40)
    for pid in range(4):
        print(f"Player {pid}: {final_scores[pid]} pts -> Rank {ranks[pid]}")
    print(f"Total kyoku: {kyoku_count}")

    print("\nScore progression:")
    print(f"{'Kyoku':>8} {'P0':>8} {'P1':>8} {'P2':>8} {'P3':>8}")
    for k in summary["kyoku"]:
        info = k["kyoku_info"]
        scores = k["end_kyoku_scores"]
        label = f"{info['bakaze']}{info['kyoku']}-{info['honba']}"
        print(f"{label:>8} {scores[0]:>8} {scores[1]:>8} {scores[2]:>8} {scores[3]:>8}")


if __name__ == "__main__":
    analyze_logs()