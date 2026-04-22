import mjai
import json
import random
from collections import Counter
from pathlib import Path

N_GAMES = 100
all_ranks = []
all_final_scores = []

for i in range(N_GAMES):
    log_dir = f"./logs/vs_shanten_{i:03d}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    seed = (random.randint(0, 2**31), random.randint(0, 2**31))

    try:
        mjai.Simulator(
            ["submission.zip", "shanten.zip", "shanten.zip", "shanten.zip"],
            logs_dir=log_dir,
            timeout=120.0,
            seed=seed,
        ).run()

        with open(f"{log_dir}/summary.json") as f:
            summary = json.load(f)

        rank = summary["rank"][0]
        score = summary["kyoku"][-1]["end_kyoku_scores"][0]
        kyoku_count = summary["kyoku_count"]
        all_ranks.append(rank)
        all_final_scores.append(score)
        print(f"Game {i+1:3d}/{N_GAMES}: Rank {rank}, Score {score:>7}, Kyoku {kyoku_count}, Seed {seed}", flush=True)

    except Exception as e:
        print(f"Game {i+1:3d}/{N_GAMES}: ERROR - {e}", flush=True)
        import subprocess
        subprocess.run(
            'docker rm -f $(docker ps -a --filter ancestor=smly/mjai-client:v3 --format "{{.ID}}" 2>/dev/null) 2>/dev/null || true',
            shell=True
        )
        continue

N = len(all_ranks)
if N == 0:
    print("No games completed.")
else:
    rank_dist = Counter(all_ranks)
    print(f"\n===== Results vs Shanten Bot ({N} games) =====")
    print(f"Avg rank:        {sum(all_ranks)/N:.2f}  (random expected: 2.50)")
    print(f"Avg final score: {sum(all_final_scores)/N:.0f}")
    print(f"Rank 1 rate:     {rank_dist[1]/N*100:.1f}%  (random expected: 25.0%)")
    print(f"Rank 2 rate:     {rank_dist[2]/N*100:.1f}%")
    print(f"Rank 3 rate:     {rank_dist[3]/N*100:.1f}%")
    print(f"Rank 4 rate:     {rank_dist[4]/N*100:.1f}%")
    print(f"Top-2 rate:      {(rank_dist[1]+rank_dist[2])/N*100:.1f}%  (random expected: 50.0%)")