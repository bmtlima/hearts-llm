from agents.base import BaseAgent
from games.hearts import suit, rank_value, RANKS


class DuckAgent(BaseAgent):
    def __init__(self):
        self.cards_played: set[str] = set()
        self.hearts_broken: bool = False

    def reset(self):
        self.cards_played = set()
        self.hearts_broken = False

    def _update_state(self, events: list[dict]):
        for e in events:
            if e["type"] == "card_played":
                card = e["card"]
                self.cards_played.add(card)
                if suit(card) == "H":
                    self.hearts_broken = True

    def _cards_in_suit(self, hand: list[str], s: str) -> list[str]:
        return [c for c in hand if suit(c) == s]

    def _current_winner_card(self, current_trick: list[tuple[int, str]]) -> str:
        led_s = suit(current_trick[0][1])
        in_suit = [(p, c) for p, c in current_trick if suit(c) == led_s]
        return max(in_suit, key=lambda pc: rank_value(pc[1]))[1]

    def choose_action(
        self,
        events_since_last_turn: list[dict],
        visible_state: dict,
        legal_actions: list,
        **kwargs,
    ) -> str:
        self._update_state(events_since_last_turn)
        hand = visible_state["your_hand"]
        trick = visible_state["current_trick"]
        trick_number = visible_state["trick_number"]

        if len(legal_actions) == 1:
            return legal_actions[0]

        # First trick special case
        if trick_number == 1:
            if not trick:
                # Leading trick 1 — must be 2C
                return "2C"
            # Following on trick 1: play highest club (safe to shed)
            return max(legal_actions, key=lambda c: rank_value(c))

        # Leading
        if not trick:
            return self._lead(legal_actions)

        # Following or void
        led_s = suit(trick[0][1])
        in_suit = self._cards_in_suit(legal_actions, led_s)

        if in_suit:
            return self._follow_suit(in_suit, trick)
        else:
            return self._void_play(legal_actions)

    def _follow_suit(self, in_suit: list[str], trick: list[tuple[int, str]]) -> str:
        winner_card = self._current_winner_card(trick)
        winner_val = rank_value(winner_card)

        # Try to duck under the current winner
        below = [c for c in in_suit if rank_value(c) < winner_val]
        if below:
            return max(below, key=lambda c: rank_value(c))

        # Can't duck — play highest (taking trick anyway)
        return max(in_suit, key=lambda c: rank_value(c))

    def _void_play(self, legal_actions: list[str]) -> str:
        # Dump QS
        if "QS" in legal_actions:
            return "QS"
        # Dump AS
        if "AS" in legal_actions:
            return "AS"
        # Dump KS
        if "KS" in legal_actions:
            return "KS"
        # Highest heart
        hearts = self._cards_in_suit(legal_actions, "H")
        if hearts:
            return max(hearts, key=lambda c: rank_value(c))
        # Highest card in any suit
        return max(legal_actions, key=lambda c: rank_value(c))

    def _lead(self, legal_actions: list[str]) -> str:
        # Lowest club, then lowest diamond
        clubs = self._cards_in_suit(legal_actions, "C")
        if clubs:
            return min(clubs, key=lambda c: rank_value(c))
        diamonds = self._cards_in_suit(legal_actions, "D")
        if diamonds:
            return min(diamonds, key=lambda c: rank_value(c))

        # Lowest spade if QS already played or we don't have it
        spades = self._cards_in_suit(legal_actions, "S")
        if spades:
            qs_played = "QS" in self.cards_played
            we_have_qs = "QS" in legal_actions
            if qs_played or not we_have_qs:
                return min(spades, key=lambda c: rank_value(c))

        # Hearts broken or only hearts
        hearts = self._cards_in_suit(legal_actions, "H")
        if hearts:
            return min(hearts, key=lambda c: rank_value(c))

        # Fallback
        return min(legal_actions, key=lambda c: rank_value(c))
