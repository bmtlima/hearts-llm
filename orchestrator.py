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
            "tricks": [...],
            "llm_turns": [...],
        }
    """
    game.deal()

    pending_events = {i: [] for i in range(4)}
    # Baselines get ALL events (including Player 0's) since they
    # don't have conversation history to track their own plays.
    baseline_pending = {}
    if baselines:
        baseline_pending = {name: [] for name in baselines}

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
            if hasattr(agent, "last_turn_metadata"):
                turn_log.update(agent.last_turn_metadata)
            llm_turns.append(turn_log)

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

        # --- If a trick just completed, record it ---
        for e in new_events:
            if e["type"] == "trick_complete":
                tricks.append(e)

    scores = game.get_scores()

    return {
        "scores": scores,
        "tricks": tricks,
        "llm_turns": llm_turns,
    }
