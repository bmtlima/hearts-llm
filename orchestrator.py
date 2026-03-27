from games.hearts import suit, sort_hand, FULL_DECK, SUITS


def compute_oracle_info(game, player_id, cards_played, player_voids):
    """Compute perfect-information stats from true game state."""
    hand = set(game.hands[player_id])
    played = set(cards_played)

    # Cards remaining: not played, not in player's hand
    remaining = set(FULL_DECK) - played - hand
    # Also exclude cards currently in the trick (they're visible but not yet "played" in our tracking)
    for _, card in game.current_trick:
        remaining.discard(card)

    remaining_by_suit = {}
    for s in SUITS:
        cards_in_suit = sort_hand([c for c in remaining if suit(c) == s])
        remaining_by_suit[s] = cards_in_suit

    # Queen of spades status
    if "QS" in hand:
        queen_status = "In your hand"
    elif "QS" in played or any(c == "QS" for _, c in game.current_trick):
        queen_status = "Already played"
    else:
        queen_status = "Still in play (not in your hand)"

    # Known voids: only report other players
    known_voids = {}
    for pid, void_suits in player_voids.items():
        if pid != player_id and void_suits:
            known_voids[pid] = void_suits

    return {
        "hearts_broken": game.hearts_broken,
        "scores": dict(game.scores),
        "remaining_by_suit": remaining_by_suit,
        "queen_status": queen_status,
        "known_voids": known_voids,
    }


def play_hand(game, agents, baselines=None, info_mode="raw", verbose=False):
    """
    Play one 13-trick hand of Hearts.

    Args:
        game: HeartsGame instance (already constructed with seed)
        agents: list of 4 agents [llm, rule, rule, rule]
        baselines: optional dict of {"name": agent} to query for comparison
                   on Player 0's turns. e.g. {"duck": DuckAgent(), "rule": RuleAgent()}
        info_mode: "raw" (default) or "oracle"

    Returns:
        {
            "scores": {0: 3, 1: 0, 2: 16, 3: 7},
            "tricks": [...],
            "llm_turns": [...],
        }
    """
    game.deal()

    pending_events = {i: [] for i in range(4)}
    baseline_pending = {}
    if baselines:
        baseline_pending = {name: [] for name in baselines}

    # Track all cards played and player voids for oracle mode
    cards_played = set()
    player_voids = {i: set() for i in range(4)}

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
                    list(baseline_pending[name]), visible, legal
                )
                baseline_pending[name] = []

        # --- Compute oracle info if needed ---
        oracle_info = None
        if current_player == 0 and info_mode == "oracle":
            oracle_info = compute_oracle_info(
                game, current_player, cards_played, player_voids
            )

        # --- Agent plays ---
        card = agent.choose_action(
            events_for_player, visible, legal, oracle_info=oracle_info
        )
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
            if oracle_info:
                # Serialize for JSON (sets -> sorted lists)
                turn_log["oracle_info"] = {
                    "hearts_broken": oracle_info["hearts_broken"],
                    "scores": oracle_info["scores"],
                    "remaining_by_suit": oracle_info["remaining_by_suit"],
                    "queen_status": oracle_info["queen_status"],
                    "known_voids": {
                        str(k): sorted(v)
                        for k, v in oracle_info["known_voids"].items()
                    },
                }
            if hasattr(agent, "last_turn_metadata"):
                turn_log.update(agent.last_turn_metadata)
            llm_turns.append(turn_log)

            if verbose:
                meta = agent.last_turn_metadata if hasattr(agent, "last_turn_metadata") else {}
                duck_match = "=" if baseline_choices.get("duck") == card else "≠"
                rule_match = "=" if baseline_choices.get("rule") == card else "≠"
                elapsed = f"{meta.get('elapsed_seconds', 0):.1f}s"
                retries = meta.get("num_retries", 0)
                in_tok = meta.get("input_tokens", 0)
                r_tok = meta.get("reasoning_tokens", 0)
                parts = [
                    f"  Trick {visible['trick_number']:>2}/13",
                    f"played={card}",
                    f"duck{duck_match}{baseline_choices.get('duck', '?')}",
                    f"rule{rule_match}{baseline_choices.get('rule', '?')}",
                    f"{elapsed}",
                    f"in={in_tok}",
                ]
                if r_tok:
                    parts.append(f"reason={r_tok}")
                if retries:
                    parts.append(f"retries={retries}")
                print("  ".join(parts), flush=True)

        # --- Apply action, distribute events ---
        new_events = game.apply_action(current_player, card)

        # Players don't receive their own events
        for i in range(4):
            if i != current_player:
                pending_events[i].extend(new_events)

        # Baselines receive ALL events unconditionally
        if baselines:
            for name in baseline_pending:
                baseline_pending[name].extend(new_events)

        # --- Track cards played and voids ---
        for e in new_events:
            if e["type"] == "card_played":
                cards_played.add(e["card"])
            elif e["type"] == "trick_complete":
                # Detect voids: first card determines led suit
                trick_cards = e["cards"]
                led_suit = suit(trick_cards[0][1])
                for pid, c in trick_cards:
                    if suit(c) != led_suit:
                        player_voids[pid].add(led_suit)
                tricks.append(e)

    scores = game.get_scores()

    return {
        "scores": scores,
        "tricks": tricks,
        "llm_turns": llm_turns,
    }
