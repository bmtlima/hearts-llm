# Hearts Context Rot Experiment — Implementation Plan

## Goal

Measure context rot in an LLM playing Hearts. One LLM player (Player 0) plays against 3 deterministic rule-based bots. The LLM receives full conversation history (no scratchpad, no tools). We measure play quality per trick to see if it degrades as context accumulates.

This is the baseline condition. Later experiments will add a text scratchpad and a code scratchpad and compare against this baseline.

---

## Architecture

Three components:

1. **HeartsGame** — pure game logic. Knows nothing about LLMs.
2. **Agents** — RandomAgent, DuckAgent (simple baseline), RuleAgent (strong baseline), LLMAgent.
3. **Orchestrator** — runs the game loop, manages events, computes baseline comparisons, logs everything.

Player setup: Player 0 is the LLM. Players 1–3 are RuleAgents (strong heuristic).

---

## Component 1: Game Engine

### `games/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseGame(ABC):
    @abstractmethod
    def get_visible_state(self, player_id: int) -> dict:
        """Return ONLY what this player can physically see right now."""
        pass

    @abstractmethod
    def get_legal_actions(self, player_id: int) -> list:
        pass

    @abstractmethod
    def apply_action(self, player_id: int, action: Any) -> list[dict]:
        """Execute action. Return list of events that occurred."""
        pass

    @abstractmethod
    def is_hand_over(self) -> bool:
        pass

    @abstractmethod
    def get_scores(self) -> dict:
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        pass
```

### `games/hearts.py`

Standard 4-player Hearts. No passing phase.

**Rules:**
- 52-card deck, 13 cards each.
- No passing. Just deal.
- Player with 2♣ leads trick 1. Must play the 2♣.
- Must follow suit if possible. If void in led suit, play anything.
- First trick: no hearts, no Q♠ unless a player has only penalty cards.
- Hearts cannot be led until broken (a heart was discarded on a prior trick).
- If a player has ONLY hearts and hearts aren't broken, they may lead a heart.
- Scoring: each heart = 1 point, Q♠ = 13 points. Total always 26.
- Shooting the moon: if one player takes all 26 points → they get 0, everyone else gets 26.

**Card representation:** String format `"{rank}{suit}"` where rank is `2-9, T, J, Q, K, A` and suit is `C, D, H, S`. Examples: `"2C"`, `"TH"`, `"QS"`, `"AD"`.

**`get_visible_state(player_id)` returns:**
```python
{
    "your_hand": ["2C", "KH", "QS", ...],     # sorted by suit then rank
    "current_trick": [("player_1", "JD"), ("player_2", "3D")],
    "trick_number": 5,                          # 1-indexed, 1 through 13
}
```

Include `trick_number` because any real player can count their pile. The LLM should not need to infer it from conversation length.

Do NOT include: scores, trick history, hearts-broken flag, cards played in previous tricks. Tracking those is the agent's job (and the core of what we're testing).

**`apply_action()` returns events:**
```python
# When a card is played:
{"type": "card_played", "player": 2, "card": "KD"}

# When a trick completes (emitted after the 4th card of each trick):
{"type": "trick_complete", "winner": 3, "trick_number": 5,
 "cards": [("player_0", "5D"), ("player_1", "TD"), ("player_2", "KD"), ("player_3", "AD")],
 "points_in_trick": 0}

# When the hand ends (emitted after last trick_complete):
{"type": "hand_over", "scores": {0: 3, 1: 0, 2: 16, 3: 7}}
```

**Dealing:** Accept a `seed` parameter. Use `random.Random(seed)` for shuffling so deals are reproducible.

**Testing:** After implementing, run 1000 hands with 4 RandomAgents. Assert that scores sum to 26 every hand (or 78 when shoot-the-moon gives 26 to 3 players). Assert no crashes. Assert first trick always starts with 2♣.

---

## Component 2: Agents

### `agents/base.py`

```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def choose_action(
        self,
        events_since_last_turn: list[dict],
        visible_state: dict,
        legal_actions: list,
    ) -> str:
        pass

    def reset(self):
        """Called between hands to clear any internal state."""
        pass
```

### `agents/random_agent.py`

Pick uniformly at random from `legal_actions`. Used only for game engine testing.

### `agents/duck_agent.py` — Simple Baseline

The "duck" heuristic. Tries to lose every trick by playing just under the current winner. This is the **lower baseline** — a competent but mindless strategy. Used only for comparison logging, not as an opponent.

The duck agent may track internal state (cards seen, hearts broken, etc.) since it's a bot.

**Rules, in priority order:**

**When following suit (you have cards in the led suit):**
1. Play the highest card in the led suit that is BELOW the current winning card (duck under it).
2. If all your cards in the led suit are above the current winner, play your HIGHEST card in that suit (you're taking the trick anyway — don't waste a low card).

**When void in the led suit (can't follow suit):**
1. Play Q♠ if you have it.
2. Play A♠ if you have it.
3. Play K♠ if you have it.
4. Play highest heart.
5. Play highest card in any suit.

**When leading a trick:**
1. Play lowest club, then lowest diamond. Prefer these suits.
2. If out of clubs and diamonds: play lowest spade (but only if Q♠ has already been played or you don't have it).
3. If hearts are broken (or you have only hearts): play lowest heart.
4. Fallback: play lowest card overall.

**First trick special case:** If leading (player has 2♣), play 2♣. If following, play highest club (safe to shed high clubs on trick 1 since no points are allowed).

Deterministic. No randomness.

### `agents/rule_agent.py` — Strong Baseline

This is the **primary opponent** (Players 1–3) and the **upper baseline** for comparison. ~15–20 rules. Tracks full internal state: cards seen, which players are void in which suits, hearts broken, Q♠ played.

**Internal state tracked (updated via events):**
- `cards_played`: set of all cards seen played
- `player_voids`: dict mapping player → set of suits they've shown void in
- `hearts_broken`: bool
- `queen_played`: bool
- `points_taken`: dict mapping player → points taken this hand

**Rules, in priority order:**

**FIRST TRICK (trick_number == 1):**
- If leading: play 2♣ (forced).
- If following: play A♣, then K♣, then Q♣ — shed high clubs safely. If no clubs, play highest non-penalty card.

**WHEN FOLLOWING SUIT:**
1. **Queen-of-spades dodge:** If spades were led and Q♠ hasn't been played yet:
   - If you have Q♠ and the current winner is K♠ or A♠: play Q♠ (let them eat it).
   - If you have cards below Q♠: play highest spade below Q♠.
   - If all your spades are Q♠ or above: play Q♠ (forced to take the risk).
2. **General duck:** Play highest card in led suit that's below the current winner.
3. **Forced win:** If you can't duck, play your highest card in the led suit.

**WHEN VOID (can't follow suit):**
1. Play Q♠ if you have it. (This is the #1 priority — dump it immediately.)
2. Play A♠ if you have it and Q♠ hasn't been played.
3. Play K♠ if you have it and Q♠ hasn't been played.
4. Play highest heart (shed points on others).
5. Play highest card in your longest suit (shed from length, preserve short suits for future voids).

**WHEN LEADING:**
1. **Smoke out Q♠:** If Q♠ hasn't been played and you DON'T hold it, and you have low spades (below Q): lead your lowest spade to try to flush it out.
2. **Safe lead:** Lead lowest card from your shortest non-heart suit (clubs or diamonds). Short suits get voided faster, giving future dump opportunities.
3. **Lead hearts:** If hearts are broken and you have only hearts, lead lowest heart.
4. **Fallback:** Lead lowest card overall.

**Void-creation tiebreaker:** When multiple cards are equally valid, prefer playing from your shortest non-heart suit. This gets you closer to creating a void.

Deterministic. No randomness.

### `agents/llm_agent.py` — No Scratchpad

The LLM agent maintains a conversation history (list of messages) across all turns within a hand. Reset between hands.

**System prompt (sent once at start of each hand):**
```
You are playing the card game Hearts as Player 0, against 3 other players.

Rules:
- 4 players, 13 tricks per hand. Standard 52-card deck.
- You must follow the suit that was led if you have cards in that suit.
- Hearts cannot be led until a heart has been discarded on a previous trick ("hearts broken").
- First trick: the player with the 2 of clubs leads it. No hearts or Queen of Spades may be played on the first trick (unless you have no other option).
- Scoring: each heart = 1 point, Queen of Spades (QS) = 13 points. Lowest score wins.
- If one player takes all 26 points, they score 0 and everyone else scores 26 ("shooting the moon").

Cards are written as rank + suit. Ranks: 2,3,4,5,6,7,8,9,T,J,Q,K,A. Suits: C,D,H,S.
Examples: "2C" = 2 of clubs, "TH" = 10 of hearts, "QS" = queen of spades.

Each turn I will tell you:
- What happened since your last turn (cards played, tricks won)
- Your current hand
- The current trick so far (if any cards have been played)
- Your legal plays

Respond with ONLY the card you want to play. Just the card code, nothing else. Example: "QS"
```

**Turn prompt (appended as user message each turn):**
```python
def build_turn_prompt(events, visible_state, legal_actions):
    parts = []

    if events:
        parts.append("Since your last turn:")
        for e in events:
            if e["type"] == "card_played":
                parts.append(f"  Player {e['player']} played {e['card']}")
            elif e["type"] == "trick_complete":
                cards_str = ", ".join(f"P{p}:{c}" for p, c in e["cards"])
                parts.append(f"  Trick {e['trick_number']} complete: {cards_str} → Player {e['winner']} wins ({e['points_in_trick']} pts)")
        parts.append("")

    parts.append(f"Trick {visible_state['trick_number']} of 13")
    parts.append(f"Your hand: {', '.join(visible_state['your_hand'])}")

    if visible_state["current_trick"]:
        trick_str = ", ".join(f"Player {p}: {c}" for p, c in visible_state["current_trick"])
        parts.append(f"Current trick: {trick_str}")
    else:
        parts.append("You are leading this trick.")

    parts.append(f"Legal plays: {', '.join(legal_actions)}")
    parts.append("")
    parts.append("Your play:")

    return "\n".join(parts)
```

**Key design note on events:** The orchestrator only delivers events for OTHER players' actions. The LLM's own plays are not echoed back via events — it must remember them from its own responses in the conversation history. This is intentional and realistic (you see other people play; you know what you played because you played it).

**Conversation flow:**
```
[system]    system prompt
[user]      turn 1 prompt (trick 1)
[assistant] "2C"
[user]      turn 2 prompt (events from players 1-3 in trick 1, trick 1 result, events from trick 2 before player 0's turn)
[assistant] "KD"
...continues for all 13 tricks (sometimes more if player 0 leads)
```

**Illegal play handling:**
1. Parse response: strip whitespace, uppercase, look for 2–3 character card code matching `[2-9TJQKA][CDHS]`.
2. If parse fails → re-prompt: `"I couldn't parse a card from your response. Legal plays: {list}. Reply with ONLY the card code."`
3. If parsed card not in legal_actions → re-prompt: `"{card} is not a legal play. Legal plays: {list}. Reply with ONLY the card code."`
4. Max 2 retries. If still fails → play random legal card, log the failure.
5. Each retry appends another user+assistant exchange to the history (this is intentional — the LLM sees its own failures).

**Reset between hands:** Clear the message history entirely. Each hand starts fresh.

**API configuration:**
- Temperature: 0 (deterministic for reproducibility)
- Max tokens: 10 (we only need a card code)
- Model: configurable via CLI arg (default `claude-sonnet-4-20250514`)

Use the Anthropic Python SDK. The LLM agent takes `model` and `api_key` as constructor args. Read `api_key` from `ANTHROPIC_API_KEY` env var by default.

---

## Component 3: Orchestrator

### `orchestrator.py`

```python
def play_hand(game, agents, baselines=None):
    """
    Play one 13-trick hand of Hearts.

    Args:
        game: HeartsGame instance (already constructed with seed)
        agents: list of 4 agents [llm, rule, rule, rule]
        baselines: optional dict of {"name": agent} to query for comparison
                   on Player 0's turns. e.g. {"duck": DuckAgent(), "rule": RuleAgent()}

    Returns:
        {
            "scores": {0: 3, 1: 0, 2: 16, 3: 7},
            "tricks": [...],  # per-trick data
            "llm_turns": [...],  # per-LLM-turn data (detailed)
        }
    """
    game.deal()

    pending_events = {i: [] for i in range(4)}
    tricks = []
    llm_turns = []

    while not game.is_hand_over():
        current_player = game.get_current_player()
        agent = agents[current_player]

        visible = game.get_visible_state(current_player)
        legal = game.get_legal_actions(current_player)
        events_for_player = list(pending_events[current_player])

        # --- Baseline comparison (Player 0 only) ---
        baseline_choices = {}
        if current_player == 0 and baselines:
            for name, baseline_agent in baselines.items():
                baseline_choices[name] = baseline_agent.choose_action(
                    events_for_player, visible, legal
                )

        # --- Agent plays ---
        card = agent.choose_action(events_for_player, visible, legal)
        pending_events[current_player] = []

        # --- Log LLM turn ---
        if current_player == 0:
            turn_log = {
                "trick_number": visible["trick_number"],
                "hand": list(visible["your_hand"]),
                "current_trick": visible["current_trick"],
                "legal_actions": list(legal),
                "card_played": card,
                "baseline_choices": baseline_choices,
                "events_received_count": len(events_for_player),
            }
            # LLM agent also provides extra metadata (tokens, retries, etc.)
            if hasattr(agent, "last_turn_metadata"):
                turn_log.update(agent.last_turn_metadata)
            llm_turns.append(turn_log)

        # --- Apply action, distribute events ---
        new_events = game.apply_action(current_player, card)
        for i in range(4):
            if i != current_player:
                pending_events[i].extend(new_events)

        # --- If a trick just completed, record it ---
        for e in new_events:
            if e["type"] == "trick_complete":
                tricks.append(e)

    # Deliver final events (hand_over) to baselines so they can reset
    # their internal state properly if needed
    scores = game.get_scores()

    return {
        "scores": scores,
        "tricks": tricks,
        "llm_turns": llm_turns,
    }
```

**Important:** The baseline agents (duck and rule) used for comparison must receive the same events as the LLM so their internal state stays synchronized. Feed them the same `events_for_player` and `visible_state`. They need to track cards played to make informed decisions. The simplest approach: give each baseline agent its own instance, call `choose_action` with the same args as the LLM gets, but also feed them events on non-LLM turns to keep their internal state updated.

Concretely: the baseline comparison agents shadow Player 0. They see everything Player 0 sees. On every turn (not just Player 0's), forward the relevant events to them so they can update their internal card-tracking state. On Player 0's turns, query them for what they'd play. This means the orchestrator needs to also forward other players' events to the baseline agents.

```python
# In the game loop, after distributing events to players:
if baselines:
    for name, b in baselines.items():
        # Baselines shadow Player 0: they see all events Player 0 would see
        if current_player != 0:
            # Player 0 would see this event next turn
            b._receive_events(new_events)  # internal state update method
```

The baseline agents should have a `_receive_events(events)` method that updates their internal tracking (cards_played, player_voids, etc.) without being asked to play. Alternatively, just track a separate pending_events buffer for each baseline and feed it in the choose_action call.

Pick whichever is cleaner. The key constraint: **baseline agents must have identical information to the LLM at the moment they're queried.**

---

## Component 4: Experiment Runner

### `run_experiment.py`

```python
import json
import time
import os
from datetime import datetime

def run_experiment(num_hands, model, seed=42):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/experiment_{timestamp}.jsonl"
    os.makedirs("logs", exist_ok=True)

    api_key = os.environ["ANTHROPIC_API_KEY"]

    # Summary accumulators
    all_results = []

    for hand_num in range(num_hands):
        game = HeartsGame(seed=seed + hand_num)

        llm_agent = LLMAgent(model=model, api_key=api_key)
        rule1, rule2, rule3 = RuleAgent(), RuleAgent(), RuleAgent()
        agents = [llm_agent, rule1, rule2, rule3]

        # Baselines for comparison
        duck_baseline = DuckAgent()
        rule_baseline = RuleAgent()
        baselines = {"duck": duck_baseline, "rule": rule_baseline}

        for a in agents:
            a.reset()
        for b in baselines.values():
            b.reset()

        result = play_hand(game, agents, baselines)

        hand_log = {
            "hand_number": hand_num,
            "seed": seed + hand_num,
            "model": model,
            "scores": result["scores"],
            "llm_score": result["scores"][0],
            "tricks": result["tricks"],
            "llm_turns": result["llm_turns"],
        }

        # Write each hand as one JSONL line
        with open(log_path, "a") as f:
            f.write(json.dumps(hand_log) + "\n")

        all_results.append(hand_log)

        print(f"Hand {hand_num}: LLM={result['scores'][0]}  "
              f"Bots={result['scores'][1]},{result['scores'][2]},{result['scores'][3]}")

    # Print summary
    print_summary(all_results)
    return all_results


def print_summary(results):
    """Print per-trick aggregates after experiment."""
    from collections import defaultdict

    trick_stats = defaultdict(lambda: {
        "count": 0,
        "duck_agreement": 0,
        "rule_agreement": 0,
        "illegal_plays": 0,
        "retries": 0,
        "total_input_tokens": 0,
    })

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
    print(f"{'Trick':>5} {'N':>4} {'Duck%':>6} {'Rule%':>6} {'Illegal':>8} {'Retries':>8} {'AvgTok':>8}")
    for tn in sorted(trick_stats.keys()):
        s = trick_stats[tn]
        n = s["count"]
        print(f"{tn:>5} {n:>4} "
              f"{100*s['duck_agreement']/n:>5.1f}% "
              f"{100*s['rule_agreement']/n:>5.1f}% "
              f"{s['illegal_plays']:>8} "
              f"{s['retries']:>8} "
              f"{s['total_input_tokens']/n:>8.0f}")

    # Overall LLM stats
    llm_scores = [h["llm_score"] for h in results]
    print(f"\nLLM avg score: {sum(llm_scores)/len(llm_scores):.1f}")
    print(f"LLM won {sum(1 for h in results if h['llm_score'] == min(h['scores'].values()))}/{len(results)} hands")
```

**CLI interface:**
```bash
python run_experiment.py --num-hands 10 --model claude-sonnet-4-20250514 --seed 42
```

Use `argparse`. Arguments:
- `--num-hands` (default 10)
- `--model` (default `claude-sonnet-4-20250514`)
- `--seed` (default 42)

---

## Component 5: LLM Agent Metadata

The LLM agent should expose a `last_turn_metadata` dict after each `choose_action` call:

```python
{
    "was_legal": True,          # did the first response parse to a legal card?
    "num_retries": 0,           # how many re-prompts were needed
    "raw_response": "QS",       # the raw text from the LLM
    "input_tokens": 850,        # from API response usage
    "output_tokens": 3,         # from API response usage
    "message_count": 14,        # total messages in conversation history
    "elapsed_seconds": 0.8,     # wall clock time for the API call(s)
}
```

---

## File Structure

```
project/
├── games/
│   ├── __init__.py
│   ├── base.py
│   └── hearts.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── random_agent.py
│   ├── duck_agent.py
│   ├── rule_agent.py
│   └── llm_agent.py
├── orchestrator.py
├── run_experiment.py
├── logs/               # created at runtime
└── tests/
    └── test_hearts.py
```

---

## Implementation Order

### Phase 1: Game engine
1. Implement `games/hearts.py`. All the rules above.
2. Implement `agents/random_agent.py`.
3. Write `tests/test_hearts.py`:
   - Run 1000 hands with 4 RandomAgents. Assert scores always sum to 26 (accounting for shoot-the-moon: 3 players get 26, one gets 0, total = 78... no, scores sum to 26 per hand normally, or 78 with shoot-the-moon since 3×26=78 and shooter gets 0). **Correction: normally scores sum to 26. With shoot-the-moon, one player gets 0 and three get 26, so total = 78. Assert `sum(scores.values()) in (26, 78)`.** Actually, think about this more carefully: shoot-the-moon means the shooter takes all 26 penalty points but then scores 0 and others get 26 each. So sum = 0+26+26+26 = 78. Yes.
   - Assert trick 1 always starts with 2♣.
   - Assert no crashes over 1000 random hands.
   - Assert each hand has exactly 13 tricks.
   - Assert no illegal plays are accepted (the game engine should validate).

### Phase 2: Bot agents
4. Implement `agents/duck_agent.py` with rules above.
5. Implement `agents/rule_agent.py` with rules above.
6. Test: run 100 hands — 1 RandomAgent vs 3 RuleAgents. RuleAgents should consistently outscore Random. Run 100 hands — 1 DuckAgent vs 3 RuleAgents. Duck should do respectably (average maybe 7–9 points per hand).

### Phase 3: Orchestrator
7. Implement `orchestrator.py` with baseline comparison logic.
8. Test end-to-end with DuckAgent as Player 0, RuleAgents as Players 1–3, DuckAgent and RuleAgent as baselines. Verify logs are generated, baseline_choices are populated, events flow correctly.

### Phase 4: LLM agent
9. Implement `agents/llm_agent.py` with conversation history, retry logic, metadata tracking.
10. Run 1 hand manually. Read through the full conversation history and verify:
    - System prompt is correct
    - Turn prompts include events, hand, trick info
    - The LLM's responses are parsed correctly
    - Retry logic works (intentionally test with a malformed response if needed)

### Phase 5: Experiment runner
11. Implement `run_experiment.py` with CLI, JSONL logging, per-trick summary.
12. Run 5 hands. Inspect the summary output and JSONL logs.

### Phase 6: Validation
13. Run 10 hands. Check:
    - Does the LLM understand the rules? Is it following suit?
    - What's the illegal play rate? Does it increase with trick number?
    - Does duck/rule agreement decrease with trick number? (This is the context rot signal.)
    - How many input tokens by trick 13?
    - What specific mistakes does it make?

---

## What the Summary Table Should Reveal

If context rot is a factor, you'll see patterns like:

| Trick | N | Duck% | Rule% | Illegal | AvgTokens |
|-------|---|-------|-------|---------|-----------|
| 1     | 10 | 90%  | 80%   | 0       | 200       |
| 2     | 10 | 85%  | 75%   | 0       | 350       |
| ...   |    |       |       |         |           |
| 12    | 10 | 55%  | 40%   | 2       | 1800      |
| 13    | 10 | 50%  | 35%   | 3       | 2000      |

Declining agreement with baselines + increasing illegal plays + growing token counts = context rot.

If agreement stays flat across tricks, context rot is NOT the dominant factor for this game length — which is also a valid finding.

---

## Notes for Later Phases

This plan is the **baseline (no scratchpad)** condition. The architecture is designed so that later experiments only require:

- **Text scratchpad:** Modify LLMAgent to append a "scratchpad" section after each turn where the LLM can write notes (cards seen, strategy, etc.). These notes persist in the conversation history.
- **Code scratchpad:** Give the LLM a tool that runs Python to compute game state (cards remaining, safe plays, etc.) and returns the result into the conversation.

The game engine, orchestrator, baselines, and logging all stay identical. Only the agent changes.