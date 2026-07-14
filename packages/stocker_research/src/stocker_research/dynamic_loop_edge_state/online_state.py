"""Robust online change-point and hierarchical payoff-state models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t as student_t


@dataclass(frozen=True)
class BOCPDSettings:
    """Configuration for a bounded Student-t Bayesian online change filter."""

    hazard_probability: float = 0.05
    degrees_of_freedom_floor: float = 5.0
    prior_mean_net_bps: float = 0.0
    prior_mean_variance_bps2: float = 22_500.0
    scale_prior_bps: float = 120.0
    outlier_clip_predictive_scales: float = 4.0
    max_run_length_sessions: int = 120
    minimum_likelihood: float = 1e-300

    def __post_init__(self) -> None:
        if not 0.0 < self.hazard_probability < 1.0:
            raise ValueError("hazard probability must be in (0, 1)")
        if self.degrees_of_freedom_floor <= 2.0:
            raise ValueError("Student-t degrees of freedom must exceed two")
        if self.prior_mean_variance_bps2 <= 0.0 or self.scale_prior_bps <= 0.0:
            raise ValueError("prior variance and scale must be positive")
        if self.outlier_clip_predictive_scales <= 0.0:
            raise ValueError("outlier clip must be positive")
        if self.max_run_length_sessions < 2:
            raise ValueError("max run length must be at least two")


@dataclass(frozen=True)
class BOCPDPosterior:
    p_change_now: float
    posterior_run_length_mean: float
    posterior_run_length_mode: int
    posterior_mean_net_bps: float
    posterior_std_net_bps: float
    p_edge_positive: float


@dataclass(frozen=True)
class EdgeForecast:
    p_change_now: float
    posterior_run_length_mean: float
    posterior_run_length_mode: float
    posterior_mean_net_bps: float
    posterior_std_net_bps: float
    posterior_lower_bound_net_bps: float
    p_edge_positive: float
    p_edge_active: float
    p_on_next: float
    p_off_next: float
    p_survive_horizon: float
    out_of_distribution_score: float


@dataclass(frozen=True)
class SupportEvidence:
    effective_sessions: float
    independent_stocks: int
    raw_fills: int
    effective_sample_size: float


@dataclass(frozen=True)
class PayoffObservation:
    cell_key: tuple[str, str, int]
    session: str
    net_payoff_bps: float
    effective_sample_size: float
    independent_stocks: tuple[str, ...]
    raw_fills: int
    availability_timestamp: pd.Timestamp

    def __post_init__(self) -> None:
        if not math.isfinite(self.net_payoff_bps):
            raise ValueError("net payoff must be finite")
        if self.effective_sample_size <= 0.0:
            raise ValueError("effective sample size must be positive")
        if not self.independent_stocks:
            raise ValueError("at least one independent stock is required")
        if self.raw_fills < len(set(self.independent_stocks)):
            raise ValueError("raw fills cannot be below independent stocks")
        if pd.Timestamp(self.availability_timestamp).tzinfo is None:
            raise ValueError("availability timestamp must be timezone-aware")


@dataclass(frozen=True)
class HierarchicalSettings:
    pooling_strength_sessions: float = 12.0
    minimum_shared_cells_per_session: int = 1
    sparse_uncertainty_inflation_bps: float = 80.0
    lower_bound_confidence: float = 0.9
    feature_logit_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pooling_strength_sessions < 0.0:
            raise ValueError("pooling strength cannot be negative")
        if self.minimum_shared_cells_per_session < 1:
            raise ValueError("minimum shared cells must be positive")
        if self.sparse_uncertainty_inflation_bps < 0.0:
            raise ValueError("uncertainty inflation cannot be negative")
        if not 0.5 < self.lower_bound_confidence < 1.0:
            raise ValueError("lower-bound confidence must be in (0.5, 1)")


class RobustBOCPD:
    """Bounded BOCPD with Student-t predictives and robust weighted updates.

    Each run-length branch carries a Normal-Inverse-Gamma posterior. The
    likelihood is Student-t and the sufficient-statistic update clips a single
    observation at a configured number of branch predictive scales.
    """

    def __init__(self, settings: BOCPDSettings) -> None:
        self.settings = settings
        self._alpha0 = settings.degrees_of_freedom_floor / 2.0
        self._beta0 = settings.scale_prior_bps**2 * max(self._alpha0 - 1.0, 0.5)
        self._kappa0 = max(
            settings.scale_prior_bps**2 / settings.prior_mean_variance_bps2,
            1e-6,
        )
        self._probabilities = np.array([1.0], dtype=float)
        self._mu = np.array([settings.prior_mean_net_bps], dtype=float)
        self._kappa = np.array([self._kappa0], dtype=float)
        self._alpha = np.array([self._alpha0], dtype=float)
        self._beta = np.array([self._beta0], dtype=float)
        self.observation_count = 0

    def _predictive(self, value: float, effective_sample_size: float) -> np.ndarray:
        weight = max(float(effective_sample_size), 1e-6)
        degrees = np.maximum(2.0 * self._alpha, self.settings.degrees_of_freedom_floor)
        scales = np.sqrt(
            np.maximum(
                self._beta * (self._kappa + weight) / (self._alpha * self._kappa * weight),
                1e-12,
            )
        )
        likelihood = student_t.pdf((value - self._mu) / scales, df=degrees) / scales
        return np.asarray(np.maximum(likelihood, self.settings.minimum_likelihood), dtype=float)

    def _prior_predictive(self, value: float, effective_sample_size: float) -> float:
        weight = max(float(effective_sample_size), 1e-6)
        degrees = max(2.0 * self._alpha0, self.settings.degrees_of_freedom_floor)
        scale = math.sqrt(
            self._beta0 * (self._kappa0 + weight) / (self._alpha0 * self._kappa0 * weight)
        )
        likelihood = float(
            student_t.pdf((value - self.settings.prior_mean_net_bps) / scale, df=degrees) / scale
        )
        return max(likelihood, self.settings.minimum_likelihood)

    def _updated_parameters(
        self,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        value: float,
        effective_sample_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weight = max(float(effective_sample_size), 1e-6)
        predictive_scale = np.sqrt(
            np.maximum(
                beta * (kappa + weight) / (alpha * kappa * weight),
                1e-12,
            )
        )
        clipped = mu + np.clip(
            value - mu,
            -self.settings.outlier_clip_predictive_scales * predictive_scale,
            self.settings.outlier_clip_predictive_scales * predictive_scale,
        )
        next_kappa = kappa + weight
        next_mu = (kappa * mu + weight * clipped) / next_kappa
        next_alpha = alpha + 0.5 * weight
        next_beta = beta + 0.5 * (kappa * weight / next_kappa) * (clipped - mu) ** 2
        return next_mu, next_kappa, next_alpha, next_beta

    def update(
        self,
        value: float,
        *,
        effective_sample_size: float = 1.0,
    ) -> BOCPDPosterior:
        if not math.isfinite(value):
            raise ValueError("observation must be finite")
        if effective_sample_size <= 0.0:
            raise ValueError("effective sample size must be positive")
        predictive = self._predictive(value, effective_sample_size)
        prior_predictive = self._prior_predictive(value, effective_sample_size)
        hazard = self.settings.hazard_probability
        unnormalised = np.concatenate(
            (
                np.array([hazard * prior_predictive]),
                self._probabilities * (1.0 - hazard) * predictive,
            )
        )
        if len(unnormalised) > self.settings.max_run_length_sessions + 1:
            tail = float(unnormalised[self.settings.max_run_length_sessions :].sum())
            unnormalised = unnormalised[: self.settings.max_run_length_sessions + 1]
            unnormalised[-1] = tail
        normaliser = float(unnormalised.sum())
        if not math.isfinite(normaliser) or normaliser <= 0.0:
            raise FloatingPointError("invalid BOCPD posterior normaliser")
        next_probabilities = unnormalised / normaliser

        prior_arrays = (
            np.array([self.settings.prior_mean_net_bps]),
            np.array([self._kappa0]),
            np.array([self._alpha0]),
            np.array([self._beta0]),
        )
        cp_parameters = self._updated_parameters(*prior_arrays, value, effective_sample_size)
        growth_parameters = self._updated_parameters(
            self._mu,
            self._kappa,
            self._alpha,
            self._beta,
            value,
            effective_sample_size,
        )
        max_length = len(next_probabilities)
        self._mu = np.concatenate((cp_parameters[0], growth_parameters[0]))[:max_length]
        self._kappa = np.concatenate((cp_parameters[1], growth_parameters[1]))[:max_length]
        self._alpha = np.concatenate((cp_parameters[2], growth_parameters[2]))[:max_length]
        self._beta = np.concatenate((cp_parameters[3], growth_parameters[3]))[:max_length]
        self._probabilities = next_probabilities
        self.observation_count += 1
        return self.snapshot()

    def snapshot(self) -> BOCPDPosterior:
        run_lengths = np.arange(len(self._probabilities), dtype=float)
        mean = float(np.dot(self._probabilities, self._mu))
        state_variances = self._beta / (np.maximum(self._alpha - 1.0, 0.5) * self._kappa)
        variance = float(
            np.dot(
                self._probabilities,
                state_variances + (self._mu - mean) ** 2,
            )
        )
        scales = np.sqrt(np.maximum(self._beta / (self._alpha * self._kappa), 1e-12))
        degrees = np.maximum(2.0 * self._alpha, self.settings.degrees_of_freedom_floor)
        positive = float(
            np.dot(
                self._probabilities,
                student_t.cdf(self._mu / scales, df=degrees),
            )
        )
        return BOCPDPosterior(
            p_change_now=float(self._probabilities[0]),
            posterior_run_length_mean=float(np.dot(self._probabilities, run_lengths)),
            posterior_run_length_mode=int(np.argmax(self._probabilities)),
            posterior_mean_net_bps=mean,
            posterior_std_net_bps=math.sqrt(max(variance, 1e-12)),
            p_edge_positive=float(np.clip(positive, 0.0, 1.0)),
        )


@dataclass
class _CellState:
    model: RobustBOCPD
    observations: list[PayoffObservation] = field(default_factory=list)


class HierarchicalPayoffModel:
    """Empirical-Bayes pooling of shared and loop/orientation BOCPD states."""

    def __init__(
        self,
        bocpd_settings: BOCPDSettings,
        hierarchy: HierarchicalSettings,
    ) -> None:
        self.bocpd_settings = bocpd_settings
        self.hierarchy = hierarchy
        self.shared = RobustBOCPD(bocpd_settings)
        self._cells: dict[tuple[str, str, int], _CellState] = {}
        self._last_session: str | None = None

    def update_session(
        self,
        session: str,
        observations: Sequence[PayoffObservation],
    ) -> None:
        if self._last_session is not None and session <= self._last_session:
            raise ValueError("sessions must update once in strictly increasing order")
        if len({item.cell_key for item in observations}) != len(observations):
            raise ValueError("duplicate cell observation in one session")
        if any(item.session != session for item in observations):
            raise ValueError("observation session does not match update session")
        if len(observations) >= self.hierarchy.minimum_shared_cells_per_session:
            values = np.asarray([item.net_payoff_bps for item in observations], dtype=float)
            lower, upper = np.quantile(values, [0.1, 0.9])
            shared_value = float(np.clip(values, lower, upper).mean())
            shared_weight = float(
                np.clip(
                    np.mean([item.effective_sample_size for item in observations]),
                    1.0,
                    20.0,
                )
            )
            self.shared.update(shared_value, effective_sample_size=shared_weight)
        for observation in observations:
            state = self._cells.setdefault(
                observation.cell_key,
                _CellState(RobustBOCPD(self.bocpd_settings)),
            )
            state.model.update(
                observation.net_payoff_bps,
                effective_sample_size=observation.effective_sample_size,
            )
            state.observations.append(observation)
        self._last_session = session

    def _support(
        self,
        state: _CellState | None,
        posterior: BOCPDPosterior | None,
    ) -> SupportEvidence:
        if state is None or posterior is None or not state.observations:
            return SupportEvidence(0.0, 0, 0, 0.0)
        effective_sessions = min(
            float(len(state.observations)),
            max(1.0, posterior.posterior_run_length_mean + 1.0),
        )
        window = min(len(state.observations), max(1, int(math.ceil(effective_sessions))))
        recent = state.observations[-window:]
        stocks = {stock for item in recent for stock in item.independent_stocks}
        fraction = effective_sessions / window
        return SupportEvidence(
            effective_sessions=effective_sessions,
            independent_stocks=len(stocks),
            raw_fills=int(round(sum(item.raw_fills for item in recent) * fraction)),
            effective_sample_size=float(
                sum(item.effective_sample_size for item in recent) * fraction
            ),
        )

    def forecast(
        self,
        cell_key: tuple[str, str, int],
        *,
        horizon_bars: int,
        session_bars: int,
        leading_features: Mapping[str, float] | None = None,
        out_of_distribution_score: float = 0.0,
        include_leading_features: bool = True,
    ) -> tuple[EdgeForecast, SupportEvidence]:
        if horizon_bars <= 0 or session_bars <= 0:
            raise ValueError("horizon and session bars must be positive")
        state = self._cells.get(cell_key)
        cell = state.model.snapshot() if state is not None else None
        shared = self.shared.snapshot()
        support = self._support(state, cell)
        evidence = support.effective_sessions
        pooling_disabled = self.hierarchy.pooling_strength_sessions == 0.0
        if pooling_disabled and cell is None:
            cell = RobustBOCPD(self.bocpd_settings).snapshot()
        if pooling_disabled:
            cell_weight = 1.0
        elif self.shared.observation_count == 0:
            cell_weight = 1.0 if cell is not None else 0.0
        else:
            cell_weight = evidence / (evidence + self.hierarchy.pooling_strength_sessions)
        if cell is None:
            cell = BOCPDPosterior(
                p_change_now=shared.p_change_now,
                posterior_run_length_mean=0.0,
                posterior_run_length_mode=0,
                posterior_mean_net_bps=shared.posterior_mean_net_bps,
                posterior_std_net_bps=shared.posterior_std_net_bps,
                p_edge_positive=shared.p_edge_positive,
            )
        mean = (
            cell_weight * cell.posterior_mean_net_bps
            + (1.0 - cell_weight) * shared.posterior_mean_net_bps
        )
        variance = (
            cell_weight**2 * cell.posterior_std_net_bps**2
            + (1.0 - cell_weight) ** 2 * shared.posterior_std_net_bps**2
            + (1.0 - cell_weight)
            * self.hierarchy.sparse_uncertainty_inflation_bps**2
            / max(evidence + 1.0, 1.0)
        )
        std = math.sqrt(max(variance, 1e-12))
        degrees = self.bocpd_settings.degrees_of_freedom_floor
        p_positive = float(student_t.cdf(mean / std, df=degrees))
        features = leading_features or {}
        feature_effect = (
            sum(
                float(self.hierarchy.feature_logit_weights.get(name, 0.0)) * float(value)
                for name, value in features.items()
                if math.isfinite(float(value))
            )
            if include_leading_features
            else 0.0
        )
        bounded_positive = float(np.clip(p_positive, 1e-9, 1.0 - 1e-9))
        p_active = float(expit(logit(bounded_positive) + feature_effect))
        decay_pressure = max(0.0, -feature_effect)
        next_hazard = float(
            np.clip(
                self.bocpd_settings.hazard_probability * math.exp(min(decay_pressure, 2.0)),
                0.001,
                0.8,
            )
        )
        new_state_positive = float(expit(feature_effect))
        p_on_next = float(
            np.clip(
                next_hazard * new_state_positive + (1.0 - next_hazard) * p_active,
                0.0,
                1.0,
            )
        )
        p_off_next = float(
            np.clip(
                next_hazard * (1.0 - new_state_positive) + (1.0 - next_hazard) * (1.0 - p_active),
                0.0,
                1.0,
            )
        )
        horizon_fraction = horizon_bars / session_bars
        survival = float((1.0 - p_off_next) ** horizon_fraction)
        quantile = float(student_t.ppf(self.hierarchy.lower_bound_confidence, df=degrees))
        return (
            EdgeForecast(
                p_change_now=float(
                    cell_weight * cell.p_change_now + (1.0 - cell_weight) * shared.p_change_now
                ),
                posterior_run_length_mean=float(
                    cell_weight * cell.posterior_run_length_mean
                    + (1.0 - cell_weight) * shared.posterior_run_length_mean
                ),
                posterior_run_length_mode=float(cell.posterior_run_length_mode),
                posterior_mean_net_bps=mean,
                posterior_std_net_bps=std,
                posterior_lower_bound_net_bps=mean - quantile * std,
                p_edge_positive=p_positive,
                p_edge_active=p_active,
                p_on_next=p_on_next,
                p_off_next=p_off_next,
                p_survive_horizon=survival,
                out_of_distribution_score=float(out_of_distribution_score),
            ),
            support,
        )
