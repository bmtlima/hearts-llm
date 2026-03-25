"""Convert a JSONL experiment log into a human-readable summary.

Usage:
    python summarize_log.py logs/experiment_raw_20260322_225951.jsonl
    python summarize_log.py logs/experiment_raw_20260322_225951.jsonl -o more_logs/deepseek_raw.md
"""

import argparse
import json
import sys
from collections import defaultdict


def summarize(log_path, output_path=None):
    with open(log_path) as f:
        results = [json.loads(line) for line in f]

    if not results:
        print("Empty log file.", file=sys.stderr)
        return

    results.sort(key=lambda h: h["hand_number"])

    model = results[0].get("model", "unknown")
    info_mode = results[0].get("info_mode", "raw")
    num_hands = len(results)

    lines = []
    lines.append(f"Model: {model}")
    lines.append(f"Info mode: {info_mode}")
    lines.append(f"Hands: {num_hands}")
    lines.append("")

    # Per-hand results
    for h in results:
        s = h["scores"]
        # scores keys may be strings from JSON
        s0 = s.get(0, s.get("0", 0))
        s1 = s.get(1, s.get("1", 0))
        s2 = s.get(2, s.get("2", 0))
        s3 = s.get(3, s.get("3", 0))
        lines.append(
            f"Hand {h['hand_number']}: LLM={s0}  Bots={s1},{s2},{s3}"
        )

    lines.append("")

    # Per-trick summary
    trick_stats = defaultdict(
        lambda: {
            "count": 0,
            "duck_agreement": 0,
            "rule_agreement": 0,
            "illegal_plays": 0,
            "retries": 0,
            "total_input_tokens": 0,
            "total_reasoning_tokens": 0,
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
            stats["total_reasoning_tokens"] += turn.get("reasoning_tokens", 0)

    has_reasoning = any(s["total_reasoning_tokens"] > 0 for s in trick_stats.values())

    header = f"{'Trick':>5} {'N':>4} {'Duck%':>6} {'Rule%':>6} {'Illegal':>8} {'Retries':>8} {'AvgTok':>8}"
    if has_reasoning:
        header += f" {'AvgReason':>10}"
    lines.append("=== Per-Trick Summary ===")
    lines.append(header)

    for tn in sorted(trick_stats.keys()):
        s = trick_stats[tn]
        n = s["count"]
        row = (
            f"{tn:>5} {n:>4} "
            f"{100*s['duck_agreement']/n:>5.1f}% "
            f"{100*s['rule_agreement']/n:>5.1f}% "
            f"{s['illegal_plays']:>8} "
            f"{s['retries']:>8} "
            f"{s['total_input_tokens']/n:>8.0f}"
        )
        if has_reasoning:
            row += f" {s['total_reasoning_tokens']/n:>10.0f}"
        lines.append(row)

    lines.append("")

    llm_scores = [h["llm_score"] for h in results]
    lines.append(f"LLM avg score: {sum(llm_scores)/len(llm_scores):.1f}")
    lines.append(
        f"LLM won {sum(1 for h in results if h['llm_score'] == min(h['scores'].values()))}"
        f"/{len(results)} hands"
    )

    text = "\n".join(lines) + "\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
        print(f"Written to {output_path}")
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment JSONL log")
    parser.add_argument("log_file", help="Path to .jsonl log file")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    args = parser.parse_args()
    summarize(args.log_file, args.output)


if __name__ == "__main__":
    main()
