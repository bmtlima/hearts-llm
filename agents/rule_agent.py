from agents.base import BaseAgent
from games.hearts import suit, rank_value, RANKS


class RuleAgent(BaseAgent):
    def __init__(self):
        self.cards_played: set[str] = set()
        self.player_voids: dict[int, set[str]] = {i: set() for i in range(4)}
        self.hearts_broken: bool = False
        self.queen_played: bool = False
        self.points_taken: dict[int, int] = {i: 0 for i in range(4)}

    def reset(self):
        self.cards_played = set()
        self.player_voids = {i: set() for i in range(4)}
        self.hearts_broken = False
        self.queen_played = False
        self.points_taken = {i: 0 for i in range(4)}

    def _update_state(self, events: list[dict]):
        for e in events:
            if e["type"] == "card_played":
                card = e["card"]
                self.cards_played.add(card)
                if suit(card) == "H":
                    self.hearts_broken = True
                if card == "QS":
                    self.queen_played = True
            elif e["type"] == "trick_complete":
                winner = e["winner"]
                self.points_taken[winner] += e["points_in_trick"]
                # Detect voids: first card determines led suit
                cards = e["cards"]
                led_s = suit(cards[0][1])
                for player, card in cards:
                    if suit(card) != led_s:
                        self.player_voids[player].add(led_s)

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

        # FIRST TRICK
        if trick_number == 1:
            return self._first_trick(trick, legal_actions)

        # Leading
        if not trick:
            return self._lead(legal_actions)

        # Following or void
        led_s = suit(trick[0][1])
        in_suit = self._cards_in_suit(legal_actions, led_s)

        if in_suit:
            return self._follow_suit(in_suit, led_s, trick)
        else:
            return self._void_play(legal_actions)

    def _first_trick(self, trick: list[tuple[int, str]], legal_actions: list[str]) -> str:
        if not trick:
            return "2C"  # forced lead
        # Following on trick 1: shed high clubs safely
        # Play AC, then KC, then QC
        for card in ["AC", "KC", "QC"]:
            if card in legal_actions:
                return card
        # Play highest available club
        clubs = self._cards_in_suit(legal_actions, "C")
        if clubs:
            return max(clubs, key=lambda c: rank_value(c))
        # Void in clubs — play highest non-penalty card
        non_penalty = [c for c in legal_actions if c != "QS" and suit(c) != "H"]
        if non_penalty:
            return max(non_penalty, key=lambda c: rank_value(c))
        # Only penalty cards (shouldn't normally happen with proper legal_actions)
        return max(legal_actions, key=lambda c: rank_value(c))

    def _follow_suit(
        self, in_suit: list[str], led_s: str, trick: list[tuple[int, str]]
    ) -> str:
        winner_card = self._current_winner_card(trick)
        winner_val = rank_value(winner_card)

        # Queen-of-spades dodge
        if led_s == "S" and not self.queen_played:
            # If we have QS and current winner is KS or AS: play QS
            if "QS" in in_suit and winner_val > rank_value("Q" + "S"):
                return "QS"
            # Play highest spade below QS
            below_q = [c for c in in_suit if rank_value(c) < rank_value("QS")]
            if below_q:
                return max(below_q, key=lambda c: rank_value(c))
            # All spades are QS or above: play QS if we have it
            if "QS" in in_suit:
                return "QS"
            # Play highest (forced)
            return max(in_suit, key=lambda c: rank_value(c))

        # General duck: play highest card below current winner
        below = [c for c in in_suit if rank_value(c) < winner_val]
        if below:
            return max(below, key=lambda c: rank_value(c))

        # Forced win: play highest
        return max(in_suit, key=lambda c: rank_value(c))

    def _void_play(self, legal_actions: list[str]) -> str:
        # 1. Dump QS
        if "QS" in legal_actions:
            return "QS"
        # 2. Dump AS if QS not played
        if not self.queen_played and "AS" in legal_actions:
            return "AS"
        # 3. Dump KS if QS not played
        if not self.queen_played and "KS" in legal_actions:
            return "KS"
        # 4. Highest heart
        hearts = self._cards_in_suit(legal_actions, "H")
        if hearts:
            return max(hearts, key=lambda c: rank_value(c))
        # 5. Highest card from longest suit
        return self._highest_from_longest(legal_actions)

    def _highest_from_longest(self, legal_actions: list[str]) -> str:
        suit_groups: dict[str, list[str]] = {}
        for c in legal_actions:
            s = suit(c)
            suit_groups.setdefault(s, []).append(c)
        # Find longest suit, break ties by preferring non-hearts, then by shortest suit name (alphabetical)
        longest = max(
            suit_groups.items(),
            key=lambda item: (len(item[1]), item[0] != "H"),
        )
        return max(longest[1], key=lambda c: rank_value(c))

    def _lead(self, legal_actions: list[str]) -> str:
        # 1. Smoke out QS: if QS not played and we don't hold it, lead low spade
        if not self.queen_played and "QS" not in legal_actions:
            spades = self._cards_in_suit(legal_actions, "S")
            low_spades = [c for c in spades if rank_value(c) < rank_value("QS")]
            if low_spades:
                return min(low_spades, key=lambda c: rank_value(c))

        # 2. Safe lead: lowest card from shortest non-heart suit
        non_heart_suits: dict[str, list[str]] = {}
        for c in legal_actions:
            s = suit(c)
            if s != "H":
                non_heart_suits.setdefault(s, []).append(c)
        if non_heart_suits:
            # Shortest non-heart suit
            shortest_suit = min(non_heart_suits.items(), key=lambda item: len(item[1]))
            return min(shortest_suit[1], key=lambda c: rank_value(c))

        # 3. Hearts broken or only hearts
        hearts = self._cards_in_suit(legal_actions, "H")
        if hearts:
            return min(hearts, key=lambda c: rank_value(c))

        # 4. Fallback
        return min(legal_actions, key=lambda c: rank_value(c))
