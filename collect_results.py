import json
from collections import Counter
from pathlib import Path

all_ranks = []
all_scores = []

for log_dir in sorted(Path("./logs").glob("vs_shanten_*")):
    summary_file = log_dir / "summary.json"
    if not summary_file.exists():
        continue
    with open(summary_file) as f:
        summary = json.load(f)
    rank = summary["rank"][0]
    score = summary["kyoku"][-1]["end_kyoku_scores"][0]
    all_ranks.append(rank)
    all_scores.append(score)

N = len(all_ranks)
rank_dist = Counter(all_ranks)
print(f"Games played: {N}")
print(f"Avg rank: {sum(all_ranks)/N:.2f}  (random expected: 2.50)")
print(f"Avg final score: {sum(all_scores)/N:.0f}")
print(f"Rank 1 rate: {rank_dist[1]/N*100:.1f}%  (random expected: 25.0%)")
print(f"Rank 2 rate: {rank_dist[2]/N*100:.1f}%")
print(f"Rank 3 rate: {rank_dist[3]/N*100:.1f}%")
print(f"Rank 4 rate: {rank_dist[4]/N*100:.1f}%")
print(f"Top-2 rate: {(rank_dist[1]+rank_dist[2])/N*100:.1f}%  (random expected: 50.0%)")