import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from agents.duck_agent import DuckAgent
from agents.llm_agent import LLMAgent
from agents.rule_agent import RuleAgent
from games.hearts import HeartsGame
from orchestrator import play_hand


def play_single_hand(hand_num, seed, model, api_key, info_mode):
    """Play one hand. Fully self-contained — no shared state."""
    game = HeartsGame(seed=seed + hand_num)

    llm_agent = LLMAgent(model=model, api_key=api_key)
    rule1, rule2, rule3 = RuleAgent(), RuleAgent(), RuleAgent()
    agents = [llm_agent, rule1, rule2, rule3]

    duck_baseline = DuckAgent()
    rule_baseline = RuleAgent()
    baselines = {"duck": duck_baseline, "rule": rule_baseline}

    result = play_hand(game, agents, baselines, info_mode=info_mode)

    return {
        "hand_number": hand_num,
        "seed": seed + hand_num,
        "model": model,
        "info_mode": info_mode,
        "scores": result["scores"],
        "llm_score": result["scores"][0],
        "tricks": result["tricks"],
        "llm_turns": result["llm_turns"],
    }


def run_experiment(num_hands, model, seed=42, info_mode="raw", workers=1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/experiment_{info_mode}_{timestamp}.jsonl"
    os.makedirs("logs", exist_ok=True)

    api_key = os.environ["OPENROUTER_API_KEY"]

    all_results = []

    if workers <= 1:
        for hand_num in range(num_hands):
            hand_log = play_single_hand(hand_num, seed, model, api_key, info_mode)
            all_results.append(hand_log)
            s = hand_log["scores"]
            print(
                f"Hand {hand_num}: LLM={s[0]}  "
                f"Bots={s[1]},{s[2]},{s[3]}"
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(play_single_hand, h, seed, model, api_key, info_mode): h
                for h in range(num_hands)
            }
            for future in as_completed(futures):
                hand_log = future.result()
                all_results.append(hand_log)
                s = hand_log["scores"]
                print(
                    f"Hand {hand_log['hand_number']}: LLM={s[0]}  "
                    f"Bots={s[1]},{s[2]},{s[3]}  "
                    f"({len(all_results)}/{num_hands})"
                )

    # Sort by hand number and write JSONL
    all_results.sort(key=lambda h: h["hand_number"])
    with open(log_path, "w") as f:
        for hand_log in all_results:
            f.write(json.dumps(hand_log) + "\n")

    print_summary(all_results)
    return all_results


def print_summary(results):
    """Print per-trick aggregates after experiment."""
    trick_stats = defaultdict(
        lambda: {
            "count": 0,
            "duck_agreement": 0,
            "rule_agreement": 0,
            "illegal_plays": 0,
            "retries": 0,
            "total_input_tokens": 0,
        }
    )

    for hand in results:
        for turn in hand["llm_turns"]:
            tn = turn["trick_number"]
            stats = trick_stats[tn]
            stats["count"] += 1
            if turn["baseline_choices"].get("duck") == turn["card_played"]:
                stats["duck_agreement"] += 1
            if turn["baseline_choices"].get("rule") == turn["card_played"]:
                stats["rule_agreement"] += 1
            stats["illegal_plays"] += 1 if not turn.get("was_legal", True) else 0
            stats["retries"] += turn.get("num_retries", 0)
            stats["total_input_tokens"] += turn.get("input_tokens", 0)

    print("\n=== Per-Trick Summary ===")
    print(
        f"{'Trick':>5} {'N':>4} {'Duck%':>6} {'Rule%':>6} "
        f"{'Illegal':>8} {'Retries':>8} {'AvgTok':>8}"
    )
    for tn in sorted(trick_stats.keys()):
        s = trick_stats[tn]
        n = s["count"]
        print(
            f"{tn:>5} {n:>4} "
            f"{100*s['duck_agreement']/n:>5.1f}% "
            f"{100*s['rule_agreement']/n:>5.1f}% "
            f"{s['illegal_plays']:>8} "
            f"{s['retries']:>8} "
            f"{s['total_input_tokens']/n:>8.0f}"
        )

    llm_scores = [h["llm_score"] for h in results]
    print(f"\nLLM avg score: {sum(llm_scores)/len(llm_scores):.1f}")
    print(
        f"LLM won {sum(1 for h in results if h['llm_score'] == min(h['scores'].values()))}"
        f"/{len(results)} hands"
    )


def main():
    parser = argparse.ArgumentParser(description="Hearts context-rot experiment")
    parser.add_argument("--num-hands", type=int, default=10)
    parser.add_argument("--model", type=str, default="anthropic/claude-haiku-4.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--info-mode",
        type=str,
        choices=["raw", "oracle", "scratchpad"],
        default="raw",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run_experiment(args.num_hands, args.model, args.seed, args.info_mode, args.workers)


if __name__ == "__main__":
    main()
