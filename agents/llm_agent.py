import random
import re
import time

from openai import OpenAI

from agents.base import BaseAgent

SYSTEM_PROMPT = """You are playing the card game Hearts as Player 0, against 3 other players.

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
"""

MAX_RETRIES = 2
CARD_PATTERN = re.compile(r"[2-9TJQKA][CDHS]")
ORACLE_BLOCK = re.compile(r"\nAdditional information:\n.*?(?=\nYour play:)", re.DOTALL)


SUIT_NAMES = {"C": "Clubs", "D": "Diamonds", "H": "Hearts", "S": "Spades"}


def build_turn_prompt(events, visible_state, legal_actions, oracle_info=None):
    parts = []

    if events:
        parts.append("Since your last turn:")
        for e in events:
            if e["type"] == "card_played":
                parts.append(f"  Player {e['player']} played {e['card']}")
            elif e["type"] == "trick_complete":
                cards_str = ", ".join(f"P{p}:{c}" for p, c in e["cards"])
                parts.append(
                    f"  Trick {e['trick_number']} complete: {cards_str} "
                    f"→ Player {e['winner']} wins ({e['points_in_trick']} pts)"
                )
        parts.append("")

    parts.append(f"Trick {visible_state['trick_number']} of 13")
    parts.append(f"Your hand: {', '.join(visible_state['your_hand'])}")

    if visible_state["current_trick"]:
        trick_str = ", ".join(
            f"Player {p}: {c}" for p, c in visible_state["current_trick"]
        )
        parts.append(f"Current trick: {trick_str}")
    else:
        parts.append("You are leading this trick.")

    parts.append(f"Legal plays: {', '.join(legal_actions)}")

    if oracle_info:
        parts.append("")
        parts.append("Additional information:")
        parts.append(f"- Hearts broken: {'Yes' if oracle_info['hearts_broken'] else 'No'}")
        scores = oracle_info["scores"]
        parts.append(
            f"- Current scores: You: {scores.get(0, 0)}, "
            f"Player 1: {scores.get(1, 0)}, "
            f"Player 2: {scores.get(2, 0)}, "
            f"Player 3: {scores.get(3, 0)}"
        )
        parts.append("- Cards still in play (not yet played, not in your hand):")
        for s in "SHDC":
            cards = oracle_info["remaining_by_suit"].get(s, [])
            name = SUIT_NAMES[s]
            if cards:
                parts.append(f"    {name}: {', '.join(cards)}")
            else:
                parts.append(f"    {name}: (none remaining)")
        parts.append(f"- Queen of spades: {oracle_info['queen_status']}")
        voids = oracle_info.get("known_voids", {})
        if voids:
            void_parts = []
            for pid in sorted(voids.keys()):
                suits = ", ".join(SUIT_NAMES[s].lower() for s in sorted(voids[pid]))
                void_parts.append(f"Player {pid} has shown no {suits}")
            parts.append(f"- Known voids: {'; '.join(void_parts)}")

    parts.append("")
    parts.append("Your play:")

    return "\n".join(parts)


class LLMAgent(BaseAgent):
    def __init__(self, model: str = "deepseek/deepseek-v3.2", api_key: str | None = None):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60.0,
        )
        self.messages: list[dict] = []
        self.last_turn_metadata: dict = {}

    def reset(self):
        self.messages = []
        self.last_turn_metadata = {}

    def choose_action(
        self,
        events_since_last_turn: list[dict],
        visible_state: dict,
        legal_actions: list,
        **kwargs,
    ) -> str:
        oracle_info = kwargs.get("oracle_info")
        prompt = build_turn_prompt(
            events_since_last_turn, visible_state, legal_actions, oracle_info
        )
        self.messages.append({"role": "user", "content": prompt})
        self._last_prompt = prompt
        self._last_api_messages = None  # set after cleaning

        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        num_retries = 0
        was_legal = True
        raw_response = ""
        reasoning_content = ""

        start_time = time.time()

        for attempt in range(1 + MAX_RETRIES):
            # Strip oracle info from old turns — only the latest message keeps it
            cleaned = []
            for msg in self.messages[:-1]:
                if msg["role"] == "user" and "\nAdditional information:\n" in msg["content"]:
                    cleaned.append({**msg, "content": ORACLE_BLOCK.sub("", msg["content"])})
                else:
                    cleaned.append(msg)
            cleaned.append(self.messages[-1])
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + cleaned
            if attempt == 0:
                self._last_api_messages = api_messages
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=10,
                temperature=0,
                messages=api_messages,
                extra_body={"reasoning": {"enabled": True, "max_tokens": 2000}},
            )

            raw_text = response.choices[0].message.content or ""
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
            # Reasoning tokens from completion_tokens_details
            ctd = response.usage.completion_tokens_details
            if ctd and ctd.reasoning_tokens:
                total_reasoning_tokens += ctd.reasoning_tokens

            # Reasoning content from model_extra
            msg = response.choices[0].message
            r_content = (msg.model_extra or {}).get("reasoning")
            if attempt == 0 and r_content:
                reasoning_content = r_content

            if attempt == 0:
                raw_response = raw_text

            # Try exact match first
            stripped = raw_text.strip().upper()
            card = None
            if CARD_PATTERN.fullmatch(stripped):
                card = stripped
            else:
                # Regex search
                match = CARD_PATTERN.search(stripped)
                if match:
                    card = match.group()

            if card is None:
                # Parse failure
                was_legal = False
                num_retries += 1
                self.messages.append({"role": "assistant", "content": raw_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"I couldn't parse a card from your response. "
                        f"Legal plays: {', '.join(legal_actions)}. "
                        f"Reply with ONLY the card code."
                    ),
                })
                continue

            if card not in legal_actions:
                # Illegal play
                was_legal = False
                num_retries += 1
                self.messages.append({"role": "assistant", "content": raw_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"{card} is not a legal play. "
                        f"Legal plays: {', '.join(legal_actions)}. "
                        f"Reply with ONLY the card code."
                    ),
                })
                continue

            # Valid card
            self.messages.append({"role": "assistant", "content": raw_text})
            elapsed = time.time() - start_time

            self.last_turn_metadata = {
                "was_legal": was_legal,
                "num_retries": num_retries,
                "raw_response": raw_response,
                "reasoning_content": reasoning_content,
                "prompt": self._last_prompt,
                "api_messages": self._last_api_messages,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "reasoning_tokens": total_reasoning_tokens,
                "message_count": len(self.messages),
                "elapsed_seconds": round(elapsed, 3),
            }
            return card

        # All retries exhausted — play random legal card
        card = random.choice(legal_actions)
        elapsed = time.time() - start_time

        self.last_turn_metadata = {
            "was_legal": False,
            "num_retries": num_retries,
            "raw_response": raw_response,
            "prompt": self._last_prompt,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "message_count": len(self.messages),
            "elapsed_seconds": round(elapsed, 3),
        }
        return card
