"""Regime stability filter — debounces fast switches and tracks flicker rate."""

import logging
from typing import List, Optional

from core.hmm.regime_state import RegimeState
from core.timeutil import utc_now

logger = logging.getLogger(__name__)


class StabilityFilter:
    """Apply ``stability_bars`` debouncing before confirming a regime switch."""

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._current_state: Optional[RegimeState] = None
        self._consecutive_bars: int = 0
        self._pending_regime_id: Optional[int] = None
        self._pending_bars: int = 0
        self._flicker_history: List[int] = []

    @property
    def current_state(self) -> Optional[RegimeState]:
        return self._current_state

    @property
    def consecutive_bars(self) -> int:
        return self._consecutive_bars

    def update(self, raw_state_id: int, probability: float, state_probs, labels: List[str]) -> RegimeState:
        """Apply stability filter and return the (possibly held) confirmed regime."""
        stability_bars = self._cfg.get("stability_bars", 3)

        if self._current_state is None:
            return self._bootstrap(raw_state_id, probability, state_probs, labels)

        if raw_state_id == self._current_state.state_id:
            return self._hold_current(probability, state_probs, labels)

        confirmed = self._try_confirm_switch(raw_state_id, probability, state_probs, labels, stability_bars)
        if confirmed is not None:
            return confirmed

        self._flicker_history.append(0)
        self._trim_window()
        return self._make_state(self._current_state.state_id, probability, state_probs, labels, is_confirmed=False)

    def get_stability(self) -> int:
        return self._consecutive_bars

    def get_flicker_rate(self) -> int:
        return sum(self._flicker_history)

    def is_flickering(self) -> bool:
        return self.get_flicker_rate() > self._cfg.get("flicker_threshold", 4)

    def _bootstrap(self, state_id: int, prob: float, state_probs, labels: List[str]) -> RegimeState:
        self._current_state = self._make_state(state_id, prob, state_probs, labels, is_confirmed=True)
        self._consecutive_bars = 1
        return self._current_state

    def _hold_current(self, prob: float, state_probs, labels: List[str]) -> RegimeState:
        self._pending_regime_id = None
        self._pending_bars = 0
        self._consecutive_bars += 1
        self._trim_window()
        return self._make_state(self._current_state.state_id, prob, state_probs, labels, is_confirmed=True)

    def _try_confirm_switch(
        self, raw_state_id: int, prob: float, state_probs, labels: List[str], stability_bars: int
    ) -> Optional[RegimeState]:
        if self._pending_regime_id == raw_state_id:
            self._pending_bars += 1
        else:
            self._pending_regime_id = raw_state_id
            self._pending_bars = 1

        if self._pending_bars < stability_bars:
            return None

        self._flicker_history.append(1)
        self._current_state = self._make_state(raw_state_id, prob, state_probs, labels, is_confirmed=True)
        self._consecutive_bars = self._pending_bars
        self._pending_regime_id = None
        self._pending_bars = 0
        logger.warning("Regime confirmed: %s (p=%.3f)", self._current_state.label, prob)
        return self._current_state

    def _make_state(self, state_id: int, prob: float, state_probs, labels: List[str], *, is_confirmed: bool) -> RegimeState:
        return RegimeState(
            label=labels[state_id],
            state_id=state_id,
            probability=prob,
            state_probabilities=state_probs,
            timestamp=utc_now(),
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
        )

    def _trim_window(self) -> None:
        window = self._cfg.get("flicker_window", 20)
        self._flicker_history = self._flicker_history[-window:]
