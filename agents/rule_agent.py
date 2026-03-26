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
                cards = e["cards"]
                led_s = suit(cards[0][1])
                for player, card in cards:
                    if suit(card) != led_s:
                        self.player_voids[player].add(led_s)

    def _detect_mid_trick_voids(self, trick: list[tuple[int, str]]):
        """Update void info from the current (incomplete) trick."""
        if not trick:
            return
        led_s = suit(trick[0][1])
        for player, card in trick[1:]:
            if suit(card) != led_s:
                self.player_voids[player].add(led_s)

    def _cards_in_suit(self, hand: list[str], s: str) -> list[str]:
        return [c for c in hand if suit(c) == s]

    def _current_winner(self, trick: list[tuple[int, str]]) -> tuple[int, str]:
        """Return (player, card) of whoever is currently winning the trick."""
        led_s = suit(trick[0][1])
        in_suit = [(p, c) for p, c in trick if suit(c) == led_s]
        return max(in_suit, key=lambda pc: rank_value(pc[1]))

    def _trick_points(self, trick: list[tuple[int, str]]) -> int:
        pts = 0
        for _, c in trick:
            if suit(c) == "H":
                pts += 1
            elif c == "QS":
                pts += 13
        return pts

    def _position_in_trick(self, trick: list[tuple[int, str]]) -> int:
        """0-indexed position: 0 = leading, 3 = last."""
        return len(trick)

    def _remaining_in_suit(self, s: str, hand: list[str]) -> list[str]:
        """Cards in suit s that haven't been played and aren't in our hand."""
        all_in_suit = [r + s for r in RANKS]
        return [c for c in all_in_suit if c not in self.cards_played and c not in hand]

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

        # Pick up any void info from the in-progress trick
        self._detect_mid_trick_voids(trick)

        if len(legal_actions) == 1:
            return legal_actions[0]

        # FIRST TRICK
        if trick_number == 1:
            return self._first_trick(trick, legal_actions)

        # Leading
        if not trick:
            return self._lead(hand, legal_actions)

        # Following or void
        led_s = suit(trick[0][1])
        in_suit = self._cards_in_suit(legal_actions, led_s)

        if in_suit:
            return self._follow_suit(in_suit, led_s, trick, hand)
        else:
            return self._void_play(legal_actions)

    # ------------------------------------------------------------------
    # FIRST TRICK
    # ------------------------------------------------------------------
    def _first_trick(
        self, trick: list[tuple[int, str]], legal_actions: list[str]
    ) -> str:
        if not trick:
            return "2C"  # forced lead

        # Following on trick 1: shed high clubs safely
        for card in ["AC", "KC", "QC"]:
            if card in legal_actions:
                return card

        clubs = self._cards_in_suit(legal_actions, "C")
        if clubs:
            return max(clubs, key=lambda c: rank_value(c))

        # Void in clubs — dump highest non-penalty card
        non_penalty = [c for c in legal_actions if c != "QS" and suit(c) != "H"]
        if non_penalty:
            return max(non_penalty, key=lambda c: rank_value(c))

        return max(legal_actions, key=lambda c: rank_value(c))

    # ------------------------------------------------------------------
    # LEADING
    # ------------------------------------------------------------------
    def _lead(self, hand: list[str], legal_actions: list[str]) -> str:
        # 1. QS still out there — try to smoke it out with spades
        if not self.queen_played:
            spades = self._cards_in_suit(legal_actions, "S")

            if "QS" in legal_actions:
                # We hold QS: lead low spades to flush AS/KS first
                low_spades = [c for c in spades if rank_value(c) < rank_value("QS")]
                if low_spades:
                    return min(low_spades, key=lambda c: rank_value(c))
                # No low spades — don't lead QS, fall through to other suits
            else:
                # We don't hold QS: lead low spade to draw it out
                low_spades = [c for c in spades if rank_value(c) < rank_value("QS")]
                if low_spades:
                    return min(low_spades, key=lambda c: rank_value(c))

        # 2. Lead from a suit where we have only low cards (safe exit)
        #    Prefer suits where remaining outstanding cards are few (less risk)
        non_heart_suits: dict[str, list[str]] = {}
        for c in legal_actions:
            s = suit(c)
            if s != "H":
                non_heart_suits.setdefault(s, []).append(c)

        if non_heart_suits:
            # Score each suit: prefer short suits with low cards
            def suit_lead_score(item):
                s, cards = item
                lowest = min(rank_value(c) for c in cards)
                outstanding = self._remaining_in_suit(s, legal_actions)
                higher_outstanding = [c for c in outstanding if rank_value(c) > lowest]
                # Prefer: many higher cards outstanding (someone else takes it),
                #         short suit length (we void it sooner),
                #         low lead card
                return (len(higher_outstanding), -len(cards), -lowest)

            best_suit = max(non_heart_suits.items(), key=suit_lead_score)
            return min(best_suit[1], key=lambda c: rank_value(c))

        # 3. Only hearts left
        hearts = self._cards_in_suit(legal_actions, "H")
        if hearts:
            return min(hearts, key=lambda c: rank_value(c))

        return min(legal_actions, key=lambda c: rank_value(c))

    # ------------------------------------------------------------------
    # FOLLOWING SUIT
    # ------------------------------------------------------------------
    def _follow_suit(
        self,
        in_suit: list[str],
        led_s: str,
        trick: list[tuple[int, str]],
        hand: list[str],
    ) -> str:
        winner_player, winner_card = self._current_winner(trick)
        winner_val = rank_value(winner_card)
        position = self._position_in_trick(trick)  # 1, 2, or 3
        trick_pts = self._trick_points(trick)

        # --- Spades with QS still live: special handling ---
        if led_s == "S" and not self.queen_played:
            return self._follow_spades_queen_live(in_suit, trick, winner_val)

        # --- Cards that duck (don't take the trick) ---
        below = [c for c in in_suit if rank_value(c) < winner_val]
        above = [c for c in in_suit if rank_value(c) >= winner_val]

        if below and not above:
            # All cards duck — play the highest (save low cards for later)
            return max(below, key=lambda c: rank_value(c))

        if below and above:
            # We can choose to duck — generally duck
            # Exception: if trick is clean (0 pts), we're last, and winning
            # lets us shed a high card cheaply, still duck to be safe
            return max(below, key=lambda c: rank_value(c))

        # --- Forced to win (all cards beat the current winner) ---
        if position == 3:
            # Last to play: we're taking it no matter what
            if trick_pts == 0:
                # Clean trick — play highest to shed a dangerous card
                return max(in_suit, key=lambda c: rank_value(c))
            else:
                # Dirty trick — play lowest winner (preserve flexibility)
                return min(in_suit, key=lambda c: rank_value(c))
        else:
            # Not last: play lowest to give later players a chance to overtake us
            return min(in_suit, key=lambda c: rank_value(c))

    def _follow_spades_queen_live(
        self, in_suit: list[str], trick: list[tuple[int, str]], winner_val: int
    ) -> str:
        """Follow spades when QS hasn't been played yet."""
        position = self._position_in_trick(trick)

        # If we have QS and current winner is KS or AS, dump it
        if "QS" in in_suit and winner_val > rank_value("QS"):
            return "QS"

        # Cards safely below the queen
        below_q = [c for c in in_suit if rank_value(c) < rank_value("QS")]
        at_or_above_q = [c for c in in_suit if rank_value(c) >= rank_value("QS")]

        if below_q:
            # Duck below the queen if possible
            below_winner = [c for c in below_q if rank_value(c) < winner_val]
            if below_winner:
                # Standard duck: highest card below winner
                return max(below_winner, key=lambda c: rank_value(c))
            # All below-Q cards beat the winner — play lowest to minimize risk
            # (someone after us might play higher or dump QS on us)
            if position < 3:
                return min(below_q, key=lambda c: rank_value(c))
            else:
                # Last position and trick is clean: shed highest below Q
                if self._trick_points(trick) == 0:
                    return max(below_q, key=lambda c: rank_value(c))
                return min(below_q, key=lambda c: rank_value(c))

        # Only QS or above — forced to play dangerous cards
        if "QS" in in_suit:
            return "QS"  # dump it and hope for the best
        return min(at_or_above_q, key=lambda c: rank_value(c))

    # ------------------------------------------------------------------
    # VOID PLAY (off-suit)
    # ------------------------------------------------------------------
    def _void_play(self, legal_actions: list[str]) -> str:
        # 1. Dump QS
        if "QS" in legal_actions:
            return "QS"

        # 2. Dump dangerous spades if QS still out
        if not self.queen_played:
            if "AS" in legal_actions:
                return "AS"
            if "KS" in legal_actions:
                return "KS"

        # 3. Dump highest heart
        hearts = self._cards_in_suit(legal_actions, "H")
        if hearts:
            return max(hearts, key=lambda c: rank_value(c))

        # 4. Shed high card from shortest non-heart suit (accelerate voiding)
        return self._highest_from_shortest(legal_actions)

    def _highest_from_shortest(self, legal_actions: list[str]) -> str:
        """Dump highest card from shortest suit to accelerate voiding."""
        suit_groups: dict[str, list[str]] = {}
        for c in legal_actions:
            s = suit(c)
            suit_groups.setdefault(s, []).append(c)

        # Shortest suit first; break ties by preferring non-hearts
        shortest = min(
            suit_groups.items(),
            key=lambda item: (len(item[1]), -(item[0] != "H")),
        )
        return max(shortest[1], key=lambda c: rank_value(c))
