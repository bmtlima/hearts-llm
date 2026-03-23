from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(
        self,
        events_since_last_turn: list[dict],
        visible_state: dict,
        legal_actions: list,
    ) -> str:
        pass

    def reset(self):
        """Called between hands to clear any internal state."""
        pass
