from abc import ABC, abstractmethod
from typing import Any


class BaseGame(ABC):
    @abstractmethod
    def deal(self):
        """Deal cards to players."""
        pass

    @abstractmethod
    def get_visible_state(self, player_id: int) -> dict:
        """Return ONLY what this player can physically see right now."""
        pass

    @abstractmethod
    def get_legal_actions(self, player_id: int) -> list:
        pass

    @abstractmethod
    def apply_action(self, player_id: int, action: Any) -> list[dict]:
        """Execute action. Return list of events that occurred."""
        pass

    @abstractmethod
    def is_hand_over(self) -> bool:
        pass

    @abstractmethod
    def get_scores(self) -> dict:
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        pass
