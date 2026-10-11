"""Toy EWMA run-to-run controller; NO connection to semiconductor equipment.

Only measured MRR from a FINISHED run is provided to the controller. It does
not receive hidden drift, process noise, metrology noise, or future samples.
This is a pedagogical controller, not a validated recipe-setting algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControllerConfig:
    target_mrr: float = 75.0
    nominal_setpoint: float = 50.0
    estimated_gain: float = 0.60  # assumed calibration, not the simulation's hidden true gain
    ewma_alpha: float = 0.35
    max_step_change: float = 1.50
    min_setpoint: float = 42.0
    max_setpoint: float = 58.0

    def __post_init__(self) -> None:
        fields = (self.target_mrr, self.nominal_setpoint, self.estimated_gain,
                  self.ewma_alpha, self.max_step_change, self.min_setpoint,
                  self.max_setpoint)
        if not all(math.isfinite(x) for x in fields):
            raise ValueError("Controller configuration must be finite")
        if self.estimated_gain <= 0:
            raise ValueError("Estimated gain must be positive for this toy plant")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("Alpha must be in (0,1]")
        if self.max_step_change <= 0 or self.min_setpoint >= self.max_setpoint:
            raise ValueError("Invalid setpoint constraints")
        if not self.min_setpoint <= self.nominal_setpoint <= self.max_setpoint:
            raise ValueError("Nominal setpoint outside constraints")


class R2REWMAController:
    """One-step-delayed controller with rate and range constraints.

    Two separate moments:
    - `initial_setpoint` is applied to first run before its measurement.
    - After observing completed run `run`, `observe_completed_run` returns
      the recipe that may be applied to run+1.  Never to the current run.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._estimated_bias = 0.0
        self._last_completed_run = 0
        self._next_setpoint = config.nominal_setpoint

    @property
    def initial_setpoint(self) -> float:
        return self.config.nominal_setpoint

    @property
    def next_setpoint(self) -> float:
        return self._next_setpoint

    @property
    def estimated_bias(self) -> float:
        return self._estimated_bias

    def observe_completed_run(self, *, run: int, measured_mrr: float,
                              applied_setpoint: float) -> dict:
        """Update using only run t's finished *measured* MRR and applied recipe.

        Measurement equation assumed by controller:
            y_t = target + estimated_gain * (u_t - nominal) + bias_t + noise
        The estimated bias is smoothed, and run t+1's recipe is constrained.
        """
        if isinstance(run, bool) or not isinstance(run, int) or run != self._last_completed_run + 1:
            raise ValueError("Run number must be next completed run in order")
        if not math.isfinite(measured_mrr) or not math.isfinite(applied_setpoint):
            raise ValueError("Non-finite measurement or applied setpoint")
        if abs(applied_setpoint - self._next_setpoint) > 1e-8:
            raise ValueError("Applied setpoint differs from previously planned recipe")
        c = self.config
        raw_bias_observation = measured_mrr - c.target_mrr - c.estimated_gain * (applied_setpoint - c.nominal_setpoint)
        self._estimated_bias = (1 - c.ewma_alpha) * self._estimated_bias + c.ewma_alpha * raw_bias_observation
        desired = c.nominal_setpoint - self._estimated_bias / c.estimated_gain
        rate_limited = min(max(desired, applied_setpoint - c.max_step_change),
                           applied_setpoint + c.max_step_change)
        bounded = min(max(rate_limited, c.min_setpoint), c.max_setpoint)
        self._next_setpoint = float(bounded)
        self._last_completed_run = run
        return {
            "completed_run": run,
            "measured_mrr_completed_run": float(measured_mrr),
            "setpoint_applied_completed_run": float(applied_setpoint),
            "raw_bias_observation": float(raw_bias_observation),
            "ewma_bias_estimate_after_measurement": float(self._estimated_bias),
            "desired_setpoint_for_next_run": float(desired),
            "setpoint_for_next_run": float(bounded),
            "step_change_for_next_run": float(bounded - applied_setpoint),
            "recipe_was_constrained": bool(abs(bounded - desired) > 1e-9),
        }
