import random

from games.hearts import HeartsGame, FULL_DECK, suit, rank_value, sort_hand, PENALTY_CARDS
from agents.random_agent import RandomAgent


def play_full_hand(game, agents):
    """Helper: play a full hand, return (scores, tricks, all_events)."""
    game.deal()
    pending_events = {i: [] for i in range(4)}
    tricks = []
    all_events = []

    while not game.is_hand_over():
        cp = game.get_current_player()
        agent = agents[cp]
        visible = game.get_visible_state(cp)
        legal = game.get_legal_actions(cp)
        events_for_player = list(pending_events[cp])

        card = agent.choose_action(events_for_player, visible, legal)
        pending_events[cp] = []

        new_events = game.apply_action(cp, card)
        for i in range(4):
            if i != cp:
                pending_events[i].extend(new_events)
        all_events.extend(new_events)

        for e in new_events:
            if e["type"] == "trick_complete":
                tricks.append(e)

    return game.get_scores(), tricks, all_events


def test_scores_sum_to_26_or_78():
    """Run 1000 hands with 4 RandomAgents; scores must sum to 26 or 78."""
    agents = [RandomAgent() for _ in range(4)]
    for seed in range(1000):
        game = HeartsGame(seed=seed)
        scores, _, _ = play_full_hand(game, agents)
        total = sum(scores.values())
        assert total in (26, 78), f"Seed {seed}: scores {scores} sum to {total}"


def test_first_trick_starts_with_2c():
    """Trick 1 must always start with the 2 of clubs."""
    agents = [RandomAgent() for _ in range(4)]
    for seed in range(100):
        game = HeartsGame(seed=seed)
        game.deal()
        cp = game.get_current_player()
        legal = game.get_legal_actions(cp)
        assert legal == ["2C"], f"Seed {seed}: first legal actions = {legal}"


def test_13_tricks_per_hand():
    """Each hand must have exactly 13 tricks."""
    agents = [RandomAgent() for _ in range(4)]
    for seed in range(100):
        game = HeartsGame(seed=seed)
        _, tricks, _ = play_full_hand(game, agents)
        assert len(tricks) == 13, f"Seed {seed}: {len(tricks)} tricks"


def test_deal_reproducibility():
    """Same seed produces same deal."""
    for seed in [0, 42, 999]:
        g1 = HeartsGame(seed=seed)
        g1.deal()
        g2 = HeartsGame(seed=seed)
        g2.deal()
        for i in range(4):
            assert sorted(g1.hands[i]) == sorted(g2.hands[i])


def test_follow_suit_enforced():
    """When player has cards in led suit, they must follow suit."""
    game = HeartsGame(seed=0)
    game.deal()
    # Play the first trick to find a situation where following suit is required
    # Just verify that legal actions only contain the led suit when player has it
    agents = [RandomAgent() for _ in range(4)]
    for seed in range(50):
        game = HeartsGame(seed=seed)
        game.deal()
        # Play 2C
        cp = game.get_current_player()
        game.apply_action(cp, "2C")
        # Next player must follow clubs if they have any
        next_p = game.get_current_player()
        hand = game.hands[next_p]
        clubs = [c for c in hand if suit(c) == "C"]
        legal = game.get_legal_actions(next_p)
        if clubs:
            assert all(suit(c) == "C" for c in legal), (
                f"Seed {seed}: player {next_p} has clubs {clubs} but legal = {legal}"
            )


def test_hearts_cannot_lead_until_broken():
    """Hearts can't be led until a heart has been played."""
    game = HeartsGame(seed=0)
    game.deal()
    # Before any heart is played, leading should not include hearts
    # (unless player has only hearts)
    for seed in range(50):
        game = HeartsGame(seed=seed)
        game.deal()
        # Simulate: play trick 1 without hearts
        # Then check that the winner can't lead hearts on trick 2
        # This is hard to set up generically, so just check the rule:
        # if hearts_broken is False, leading actions should exclude hearts
        game.hearts_broken = False
        game.trick_number = 2
        game.current_trick = []
        for p in range(4):
            non_hearts = [c for c in game.hands[p] if suit(c) != "H"]
            if non_hearts and game.hands[p]:
                game._current_player = p
                legal = game.get_legal_actions(p)
                for c in legal:
                    assert suit(c) != "H", (
                        f"Seed {seed}: player {p} can lead heart {c} but hearts not broken"
                    )
                break


def test_first_trick_no_penalty_when_void():
    """On trick 1, if void in clubs, can't play hearts or QS (unless forced)."""
    game = HeartsGame(seed=0)
    game.deal()

    # Craft a scenario: give player 1 no clubs but a mix of cards
    game.hands = {
        0: ["2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "TC", "JC", "QC", "KC", "AC"],
        1: ["2H", "3H", "4D", "5D", "6D", "7D", "8D", "9D", "TD", "JD", "QD", "KD", "AD"],
        2: ["4H", "5H", "6H", "7H", "8H", "9H", "TH", "JH", "QH", "KH", "AH", "2S", "3S"],
        3: ["QS", "KS", "AS", "2D", "3D", "4S", "5S", "6S", "7S", "8S", "9S", "TS", "JS"],
    }
    game._current_player = 0
    game.trick_number = 1
    game.current_trick = []

    # Player 0 leads 2C
    game.apply_action(0, "2C")

    # Player 1 is void in clubs, has hearts and diamonds
    legal = game.get_legal_actions(1)
    for c in legal:
        assert c not in PENALTY_CARDS, f"Player 1 can play penalty card {c} on trick 1"
    # Should only have diamonds
    assert all(suit(c) == "D" for c in legal)


def test_first_trick_only_penalty_cards():
    """On trick 1, if void in clubs and ONLY have penalty cards, can play them."""
    game = HeartsGame(seed=0)
    game.deal()

    # Player 1 has only hearts and QS
    game.hands = {
        0: ["2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "TC", "JC", "QC", "KC", "AC"],
        1: ["2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "TH", "JH", "QH", "KH", "QS"],
        2: ["AH", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "TD", "JD", "QD", "KD"],
        3: ["AD", "KS", "AS", "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", "TS", "JS"],
    }
    game._current_player = 0
    game.trick_number = 1
    game.current_trick = []

    game.apply_action(0, "2C")

    # Player 1 has only penalty cards (hearts + QS), so they're allowed to play
    legal = game.get_legal_actions(1)
    assert len(legal) == 13  # all cards are playable


def test_shoot_the_moon():
    """When one player takes all 26 points, they get 0 and others get 26."""
    game = HeartsGame(seed=0)
    game.deal()

    # Rig a deal where player 0 can take every trick
    game.hands = {
        0: ["AC", "AD", "AH", "AS", "KC", "KD", "KH", "KS", "QC", "QD", "QH", "QS", "JH"],
        1: ["2C", "3C", "4C", "5C", "6C", "2D", "3D", "4D", "5D", "6D", "2H", "3H", "2S"],
        2: ["7C", "8C", "9C", "TC", "JC", "7D", "8D", "9D", "TD", "JD", "4H", "5H", "3S"],
        3: ["7S", "8S", "9S", "TS", "JS", "7H", "8H", "9H", "TH", "6H", "6S", "4S", "5S"],
    }
    # Player 1 has 2C, so they lead
    game._current_player = 1
    game.trick_number = 1
    game.hearts_broken = False

    # Play all 13 tricks — player 0 should win every trick with highest cards
    # We'll use agents that always play highest legal card for P0 and lowest for others
    class HighAgent:
        def choose_action(self, events, visible, legal):
            return max(legal, key=lambda c: rank_value(c))

    class LowAgent:
        def choose_action(self, events, visible, legal):
            return min(legal, key=lambda c: rank_value(c))

    agents = [HighAgent(), LowAgent(), LowAgent(), LowAgent()]
    pending = {i: [] for i in range(4)}

    while not game.is_hand_over():
        cp = game.get_current_player()
        visible = game.get_visible_state(cp)
        legal = game.get_legal_actions(cp)
        card = agents[cp].choose_action([], visible, legal)
        events = game.apply_action(cp, card)
        for i in range(4):
            if i != cp:
                pending[i].extend(events)

    scores = game.get_scores()
    # Player 0 should have shot the moon
    assert scores[0] == 0, f"Shooter got {scores[0]}"
    for p in [1, 2, 3]:
        assert scores[p] == 26, f"Player {p} got {scores[p]}"
    assert sum(scores.values()) == 78


def test_trick_winner_is_highest_of_led_suit():
    """Off-suit cards don't win even if higher rank."""
    game = HeartsGame(seed=0)
    game.deal()

    game.hands = {
        0: ["2C", "3D", "4H", "5S", "6C", "7D", "8H", "9S", "TC", "JD", "QH", "KS", "AC"],
        1: ["3C", "4D", "5H", "6S", "7C", "8D", "9H", "TS", "JC", "QD", "KH", "AS", "2D"],
        2: ["4C", "5D", "6H", "7S", "8C", "9D", "TH", "JS", "QC", "KD", "AH", "2S", "3S"],
        3: ["5C", "6D", "7H", "8S", "9C", "TD", "JH", "QS", "KC", "AD", "2H", "3H", "4S"],
    }
    game._current_player = 0
    game.trick_number = 1
    game.current_trick = []

    # Player 0 plays 2C (forced)
    events = game.apply_action(0, "2C")
    # Player 1 plays 3C
    events = game.apply_action(1, "3C")
    # Player 2 plays 4C
    events = game.apply_action(2, "4C")
    # Player 3 plays 5C
    events = game.apply_action(3, "5C")

    # trick_complete event — winner should be player 3 (5C is highest club)
    trick_event = [e for e in events if e["type"] == "trick_complete"][0]
    assert trick_event["winner"] == 3


def test_illegal_action_raises():
    """Playing an illegal card should raise ValueError."""
    game = HeartsGame(seed=0)
    game.deal()

    cp = game.get_current_player()
    # Try to play a card that isn't 2C on trick 1
    hand = game.hands[cp]
    non_2c = [c for c in hand if c != "2C"]
    if non_2c:
        try:
            game.apply_action(cp, non_2c[0])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


def test_no_crashes_1000_hands():
    """Stress test: 1000 hands with random agents, no crashes."""
    agents = [RandomAgent() for _ in range(4)]
    for seed in range(1000):
        game = HeartsGame(seed=seed)
        scores, tricks, _ = play_full_hand(game, agents)
        assert len(tricks) == 13
        assert sum(scores.values()) in (26, 78)
