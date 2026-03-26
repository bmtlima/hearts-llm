import random
from typing import Any

from games.base import BaseGame

RANKS = "23456789TJQKA"
SUITS = "CDHS"
FULL_DECK = [r + s for s in SUITS for r in RANKS]

PENALTY_CARDS = {"QS"} | {r + "H" for r in RANKS}


def rank_value(card: str) -> int:
    return RANKS.index(card[0])


def suit(card: str) -> str:
    return card[1]


def sort_hand(hand: list[str]) -> list[str]:
    return sorted(hand, key=lambda c: (SUITS.index(c[1]), RANKS.index(c[0])))


def points_for_card(card: str) -> int:
    if card == "QS":
        return 13
    if suit(card) == "H":
        return 1
    return 0


class HeartsGame(BaseGame):
    def __init__(self, seed: int = 0, shoot_the_moon: bool = False):
        self.shoot_the_moon = shoot_the_moon
        self.rng = random.Random(seed)
        self.hands: dict[int, list[str]] = {}
        self.current_trick: list[tuple[int, str]] = []
        self.trick_number: int = 1
        self.hearts_broken: bool = False
        self._current_player: int = 0
        self.scores: dict[int, int] = {i: 0 for i in range(4)}
        self._hand_over: bool = False
        self._tricks_played: int = 0

    def deal(self):
        deck = list(FULL_DECK)
        self.rng.shuffle(deck)
        for i in range(4):
            self.hands[i] = deck[i * 13 : (i + 1) * 13]
        self.current_trick = []
        self.trick_number = 1
        self.hearts_broken = False
        self.scores = {i: 0 for i in range(4)}
        self._hand_over = False
        self._tricks_played = 0

        # Find who has 2C — they lead
        for i in range(4):
            if "2C" in self.hands[i]:
                self._current_player = i
                break

    def get_visible_state(self, player_id: int) -> dict:
        return {
            "your_hand": sort_hand(list(self.hands[player_id])),
            "current_trick": list(self.current_trick),
            "trick_number": self.trick_number,
        }

    def get_legal_actions(self, player_id: int) -> list[str]:
        hand = self.hands[player_id]

        # Trick 1, leading: must play 2C
        if self.trick_number == 1 and len(self.current_trick) == 0:
            return ["2C"]

        # If following, must follow suit if possible
        if self.current_trick:
            led_suit = suit(self.current_trick[0][1])
            in_suit = [c for c in hand if suit(c) == led_suit]

            if in_suit:
                # First trick: can't play penalty cards even when following suit?
                # No — if you have the led suit, you must follow suit, full stop.
                # The first-trick restriction only applies when you CAN'T follow suit.
                return sort_hand(in_suit)

            # Void in led suit — can play anything, but first trick restrictions apply
            if self.trick_number == 1:
                non_penalty = [c for c in hand if c not in PENALTY_CARDS]
                if non_penalty:
                    return sort_hand(non_penalty)
                # Only penalty cards left — can play anything
                return sort_hand(list(hand))

            return sort_hand(list(hand))

        # Leading (not trick 1)
        if not self.hearts_broken:
            non_hearts = [c for c in hand if suit(c) != "H"]
            if non_hearts:
                return sort_hand(non_hearts)
            # Only hearts left — can lead a heart
        return sort_hand(list(hand))

    def apply_action(self, player_id: int, action: Any) -> list[dict]:
        legal = self.get_legal_actions(player_id)
        if action not in legal:
            raise ValueError(
                f"Illegal action {action} by player {player_id}. Legal: {legal}"
            )

        self.hands[player_id].remove(action)
        self.current_trick.append((player_id, action))

        # Track hearts broken
        if suit(action) == "H":
            self.hearts_broken = True

        events: list[dict] = [
            {"type": "card_played", "player": player_id, "card": action}
        ]

        # Check if trick is complete (4 cards played)
        if len(self.current_trick) == 4:
            led_suit = suit(self.current_trick[0][1])
            # Winner is highest card of the led suit
            winner = max(
                ((p, c) for p, c in self.current_trick if suit(c) == led_suit),
                key=lambda pc: rank_value(pc[1]),
            )[0]

            trick_points = sum(points_for_card(c) for _, c in self.current_trick)
            self.scores[winner] += trick_points

            trick_event = {
                "type": "trick_complete",
                "winner": winner,
                "trick_number": self.trick_number,
                "cards": list(self.current_trick),
                "points_in_trick": trick_points,
            }
            events.append(trick_event)

            self._tricks_played += 1
            self.current_trick = []

            if self._tricks_played == 13:
                if self.shoot_the_moon:
                    for p in range(4):
                        if self.scores[p] == 26:
                            self.scores[p] = 0
                            for other in range(4):
                                if other != p:
                                    self.scores[other] = 26
                            break
                self._hand_over = True
                events.append(
                    {"type": "hand_over", "scores": dict(self.scores)}
                )
            else:
                self.trick_number += 1
                self._current_player = winner
        else:
            # Advance to next player
            self._current_player = (self._current_player + 1) % 4

        return events

    def is_hand_over(self) -> bool:
        return self._hand_over

    def get_scores(self) -> dict:
        return dict(self.scores)

    def get_current_player(self) -> int:
        return self._current_player
