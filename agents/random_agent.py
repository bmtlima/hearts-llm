import random

from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def choose_action(
        self,
        events_since_last_turn: list[dict],
        visible_state: dict,
        legal_actions: list,
        **kwargs,
    ) -> str:
        return random.choice(legal_actions)
