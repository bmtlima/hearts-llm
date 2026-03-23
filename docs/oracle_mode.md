# Task: Add Configurable Information Modes to the LLM Agent

## Context

We have a working Hearts game where an LLM (Player 0) plays against 3 rule-based bots. Currently the LLM gets raw events + visible state (hand + current trick) and the conversation history. It averages 8.3 points per hand over 50 hands.

We want to test whether giving the LLM more information improves play. This will tell us whether the model's mistakes come from poor information tracking (fixable with a scratchpad) or poor strategy (not fixable with a scratchpad).

## What to Build

Refactor the LLM agent so it has a configurable `info_mode` parameter that controls what information gets included in the turn prompt. The conversation history should work the same way in all modes. The only thing that changes is what goes into each turn's user message.

### Mode: `raw` (what we already have)

The turn prompt includes only:
- Events since last turn
- Visible state (hand + current trick)
- Legal plays

This is the current behavior. Don't break it.

### Mode: `oracle`

The turn prompt includes everything from `raw`, PLUS computed stats that a perfect card tracker would know. These stats should be computed by the orchestrator from the true game state (the orchestrator knows everything) and passed to the agent.

The oracle stats to include in the prompt:

```
Additional information:
- Trick number: 7 of 13
- Hearts broken: Yes
- Current scores: You: 3, Player 1: 5, Player 2: 13, Player 3: 0
- Cards still in play (not yet played, not in your hand):
    Spades: KS, JS, 9S, 7S, 4S
    Hearts: AH, QH, TH, 8H, 5H
    Diamonds: (none remaining)
    Clubs: AC, TC, 6C, 3C
- Queen of spades: Still in play (not in your hand)
- Known voids: Player 1 has shown no diamonds, Player 3 has shown no clubs
```

The orchestrator needs to track the full game state (all cards played, all voids revealed) to compute these. The LLM agent itself doesn't track anything — it just receives the computed stats in its prompt.

### Mode: `scratchpad` (stub for later)

Don't implement this yet. Just make sure the architecture supports adding it later. A placeholder that behaves the same as `raw` is fine.

## Architecture

The key design decision: the **agent** shouldn't compute oracle stats. The **orchestrator** should compute them from the true game state and pass them to the agent. This keeps the agent simple and makes it easy to swap modes.

Suggested approach:

```python
# The orchestrator computes oracle info from true game state
oracle_info = compute_oracle_info(game, current_player) if info_mode == "oracle" else None

# The agent receives it as an optional parameter
card = agent.choose_action(
    events_since_last_turn=pending_events[current_player],
    visible_state=visible,
    legal_actions=legal,
    oracle_info=oracle_info,  # None for raw mode, dict for oracle mode
)
```

The `compute_oracle_info` function needs access to the game's internal state to figure out:
- What trick number we're on
- Whether hearts are broken
- Current scores per player
- All cards played so far (to compute what's remaining)
- Which players have failed to follow suit in which suits (to compute known voids)

The agent then just formats whatever it receives into the prompt. In `raw` mode, oracle_info is None and nothing extra is added. In `oracle` mode, it formats the dict into the text block shown above.

## Changes Needed

1. **Add an `info_mode` parameter** to the LLM agent (`"raw"`, `"oracle"`, `"scratchpad"`). 

2. **Add a `compute_oracle_info(game, player_id)` function** that extracts the oracle stats from the game engine's internal state. This function needs the game to expose some internals — either add getter methods to the game or have the orchestrator track game history itself.

3. **Track voids in the orchestrator or game.** When a player fails to follow the led suit, that reveals a void. The game engine might already know this implicitly (it validates legal plays), but you need to track it explicitly to report it in oracle info. The simplest approach: the orchestrator watches events and tracks voids in a dict like `{player_id: set_of_void_suits}`.

4. **Update the agent's prompt building** to include oracle info when present.

5. **Update run_experiment.py** to accept an `--info-mode` flag so you can run:
   ```bash
   python run_experiment.py --num-hands 50 --seed 42 --info-mode raw
   python run_experiment.py --num-hands 50 --seed 42 --info-mode oracle
   ```
   Same seed means same deals, so the comparison is fair.

6. **Log the info_mode** in the experiment output so results are clearly labeled.

## Important

- Use the EXACT same seed and deals for both modes. The only thing that should change between runs is what's in the prompt.
- Don't change the system prompt between modes. The system prompt explains the rules of Hearts. The oracle info is added to each turn's user message, not the system prompt.
- The oracle info should be presented plainly, not with instructions like "use this to play better." Just present it as factual information. The model should figure out how to use it.
- Keep the conversation history behavior identical across modes.
- Make sure the existing raw mode still works exactly as before — don't regress the baseline.