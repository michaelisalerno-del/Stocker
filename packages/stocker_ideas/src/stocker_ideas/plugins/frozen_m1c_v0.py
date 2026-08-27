"""Pure, no-fit calculations shared by the frozen M1C method family."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final, cast

from stocker_ideas.plugins.frozen_m1c_v0_data import DATA

M1C_THRESHOLD: Final[float] = 0.488333710794033
MINIMUM_EPISODE_SPACING_MINUTES: Final[int] = 30
EPSILON: Final[float] = 1e-12
CAUSAL_GROUP_I_FEATURES: Final[tuple[str, ...]] = (
    "arousal",
    "conviction",
    "prior_6_mean_range",
    "prior_6_price_travel",
    "prior_6_absolute_net_movement",
    "prior_6_activity_proxy",
    "recent_vs_earlier_range_ratio",
    "recent_vs_earlier_activity_ratio",
    "current_bar_range_vs_prior_6",
    "current_bar_activity_vs_prior_6",
    "current_bar_body_fraction",
    "current_bar_extreme_wick_fraction",
)
_AROUSAL_COMPONENTS = ("activity_effort", "range_effort", "travel_effort")
_CONVICTION_COMPONENTS = (
    "absolute_efficiency",
    "close_retention",
    "directional_persistence",
)
_COMPONENTS = (*_AROUSAL_COMPONENTS, *_CONVICTION_COMPONENTS)
_LOCAL_FEATURES = CAUSAL_GROUP_I_FEATURES[2:]
_LABELS = {
    "A1": "prospective hypothesis — not validated",
    "C1": "comparison only — not validated",
    "R1": "comparison only — not validated",
}
ARTIFACT_HASHES: Final[Mapping[str, str]] = cast(Mapping[str, str], DATA["artifact_hashes"])
CHECKPOINTS: Final[tuple[int, ...]] = tuple(int(value) for value in DATA["checkpoints"])
COHORT: Final[tuple[str, ...]] = tuple(str(value) for value in DATA["cohort"])
_M1C_SPEC = cast(Mapping[str, Any], DATA["m1c_feature"])["model_specification"]
_M1C_FEATURES: Final[tuple[str, ...]] = tuple(str(value) for value in _M1C_SPEC["numeric_features"])
REQUIRED_GROUP_O_FEATURES: Final[tuple[str, ...]] = tuple(
    name
    for name in _M1C_FEATURES
    if name not in CAUSAL_GROUP_I_FEATURES and not name.startswith("checkpoint_")
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be an array")
    return cast(Sequence[Any], value)


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _required_number(value: object, name: str) -> float:
    number = _finite(value)
    if number is None:
        raise ValueError(f"{name} must be finite")
    return number


def _sigmoid(linear: float) -> float:
    if linear >= 0.0:
        return 1.0 / (1.0 + math.exp(-linear))
    exponential = math.exp(linear)
    return exponential / (1.0 + exponential)


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def score_m1c(
    *,
    symbol: str,
    checkpoint: int,
    group_o: Mapping[str, object],
    group_i: Mapping[str, object],
) -> dict[str, object]:
    """Apply the exact committed M1C imputation, scaling, and logistic score."""

    stock_levels = tuple(str(value) for value in _M1C_SPEC["category_levels"]["stock"])
    if symbol not in stock_levels:
        raise ValueError("symbol is outside the frozen M1C cohort")
    if checkpoint not in CHECKPOINTS:
        raise ValueError("checkpoint is outside the frozen M1C grid")
    checkpoint_name = f"checkpoint_{checkpoint}"
    raw_values: list[float | None] = []
    for name in _M1C_FEATURES:
        if name in CAUSAL_GROUP_I_FEATURES:
            raw = group_i.get(name)
        elif name.startswith("checkpoint_"):
            raw = 1.0 if name == checkpoint_name else 0.0
        else:
            raw = group_o.get(name)
        raw_values.append(_finite(raw))
    medians = tuple(float(value) for value in _M1C_SPEC["numeric_medians"])
    means = tuple(float(value) for value in _M1C_SPEC["numeric_means"])
    scales = tuple(float(value) for value in _M1C_SPEC["numeric_scales"])
    if not (len(raw_values) == len(medians) == len(means) == len(scales)) or any(
        not math.isfinite(scale) or scale <= 0.0 for scale in scales
    ):
        raise ValueError("frozen M1C preprocessing width differs")
    transformed = tuple(
        ((medians[index] if value is None else value) - means[index]) / scales[index]
        for index, value in enumerate(raw_values)
    )
    design = (
        *transformed,
        *(float(symbol == stock) for stock in stock_levels[1:]),
    )
    coefficients = tuple(float(value) for value in _M1C_SPEC["coefficients"])
    if len(design) != len(coefficients):
        raise ValueError("frozen M1C design width differs")
    linear = math.fsum(
        value * coefficient for value, coefficient in zip(design, coefficients, strict=True)
    )
    linear += float(_M1C_SPEC["intercept"])
    probability = _sigmoid(linear)
    return {
        "model_id": "M1C",
        "model_hash": ARTIFACT_HASHES["m1c_feature"],
        "probability": probability,
        "threshold": M1C_THRESHOLD,
        "threshold_passed": probability >= M1C_THRESHOLD,
        "feature_hash": _hash_json(
            {
                "symbol": symbol,
                "checkpoint": checkpoint,
                "feature_order": _M1C_FEATURES,
                "feature_values": raw_values,
            }
        ),
        "missing_feature_count": sum(value is None for value in raw_values),
    }


def build_causal_group_i(prefix: Mapping[str, object], *, checkpoint: int) -> dict[str, float]:
    """Build the exact causal Group-I fields from one bounded session-prefix receipt."""

    if checkpoint not in CHECKPOINTS or prefix.get("bar_number") != checkpoint:
        raise ValueError("prefix does not match the frozen checkpoint")
    if prefix.get("source_completeness") != "complete":
        raise ValueError("M1C requires a complete session prefix")
    accumulator = _mapping(prefix.get("accumulator"), "prefix accumulator")
    trailing = tuple(
        _mapping(item, "trailing bar")
        for item in _sequence(prefix.get("trailing_bars"), "trailing bars")
    )
    if len(trailing) < 6:
        raise ValueError("M1C requires six trailing bars")
    activity_sum = _required_number(accumulator.get("activity_sum"), "activity sum")
    range_sum = _required_number(accumulator.get("range_sum"), "range sum")
    travel_sum = _required_number(accumulator.get("travel_sum"), "travel sum")
    return_sum = _required_number(accumulator.get("return_sum"), "return sum")
    session_open = _required_number(accumulator.get("session_open"), "session open")
    last_close = _required_number(accumulator.get("last_close"), "last close")
    width_sum = _required_number(accumulator.get("width_sum"), "width sum")
    cumulative = 10_000.0 * (last_close / session_open - 1.0)
    if abs(cumulative) <= EPSILON:
        persistence = 0.5
    else:
        count_name = "positive_return_count" if cumulative > 0.0 else "negative_return_count"
        count = accumulator.get(count_name)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("prefix directional count is invalid")
        persistence = count / checkpoint
    raw_components = {
        "activity_effort": math.log1p(activity_sum / checkpoint),
        "range_effort": math.log1p(range_sum),
        "travel_effort": math.log1p(travel_sum),
        "absolute_efficiency": abs(return_sum / max(travel_sum, EPSILON)),
        "close_retention": abs(last_close - session_open) / max(width_sum, EPSILON),
        "directional_persistence": persistence,
    }
    recent = trailing[-6:]
    ranges = tuple(
        _required_number(item.get("true_range_bps"), "trailing true range") for item in recent
    )
    returns = tuple(_required_number(item.get("return_bps"), "trailing return") for item in recent)
    activity = tuple(
        _required_number(item.get("historical_relative_activity"), "trailing activity")
        for item in recent
    )
    mean_range = math.fsum(ranges) / 6.0
    mean_activity = math.fsum(activity) / 6.0
    current = recent[-1]
    high = _required_number(current.get("high"), "current high")
    low = _required_number(current.get("low"), "current low")
    opening = _required_number(current.get("open"), "current open")
    close = _required_number(current.get("close"), "current close")
    width = high - low
    raw_local = {
        "prior_6_mean_range": mean_range,
        "prior_6_price_travel": math.fsum(abs(value) for value in returns),
        "prior_6_absolute_net_movement": abs(math.fsum(returns)),
        "prior_6_activity_proxy": mean_activity,
        "recent_vs_earlier_range_ratio": (math.fsum(ranges[3:]) / 3.0)
        / max(math.fsum(ranges[:3]) / 3.0, EPSILON),
        "recent_vs_earlier_activity_ratio": (math.fsum(activity[3:]) / 3.0)
        / max(math.fsum(activity[:3]) / 3.0, EPSILON),
        "current_bar_range_vs_prior_6": ranges[-1] / max(mean_range, EPSILON),
        "current_bar_activity_vs_prior_6": activity[-1] / max(mean_activity, EPSILON),
        "current_bar_body_fraction": min(max(abs(close - opening) / max(width, EPSILON), 0.0), 1.0),
        "current_bar_extreme_wick_fraction": max(
            _required_number(current.get("upper_wick_fraction"), "upper wick"),
            _required_number(current.get("lower_wick_fraction"), "lower wick"),
        ),
    }
    component_scaling = _mapping(
        cast(Mapping[str, object], DATA["group_i_component_scaling"])[str(checkpoint)],
        "component scaling",
    )
    scaled_components: dict[str, float] = {}
    for name in _COMPONENTS:
        fitted = _mapping(component_scaling[name], "component scale")
        center = _required_number(fitted.get("center"), "component center")
        scale = _required_number(fitted.get("scale"), "component scale")
        scaled_components[name] = min(
            max(
                (raw_components[name] - center) / scale,
                float(fitted.get("clip_lower", -5.0)),
            ),
            float(fitted.get("clip_upper", 5.0)),
        )
    local_scaling = _mapping(
        cast(Mapping[str, object], DATA["group_i_local_scaling"]).get(
            f"{prefix.get('instrument_id', '')}|{checkpoint}", {}
        ),
        "local scaling",
    )
    # Stored receipts are broker-neutral and do not repeat instrument identity in the
    # payload. Callers may provide it explicitly, otherwise use the separately supplied
    # symbol wrapper through build_group_i_for_symbol().
    if not local_scaling:
        raise ValueError("prefix payload requires an instrument_id for frozen local scaling")
    result = {
        "arousal": math.fsum(scaled_components[name] for name in _AROUSAL_COMPONENTS)
        / len(_AROUSAL_COMPONENTS),
        "conviction": math.fsum(scaled_components[name] for name in _CONVICTION_COMPONENTS)
        / len(_CONVICTION_COMPONENTS),
    }
    for name in _LOCAL_FEATURES:
        fitted = _mapping(local_scaling[name], "local feature scaling")
        result[name] = min(
            max(
                (raw_local[name] - _required_number(fitted.get("center"), "local center"))
                / _required_number(fitted.get("scale"), "local scale"),
                -5.0,
            ),
            5.0,
        )
    return result


def build_group_i_for_symbol(
    prefix: Mapping[str, object], *, symbol: str, checkpoint: int
) -> dict[str, float]:
    """Attach core-owned instrument identity before applying local frozen scaling."""

    if symbol not in COHORT:
        raise ValueError("symbol is outside the frozen M1C cohort")
    return build_causal_group_i({**prefix, "instrument_id": symbol}, checkpoint=checkpoint)


def _z(value: float, fitted: Mapping[str, Any]) -> float:
    return (value - _required_number(fitted.get("center"), "front center")) / _required_number(
        fitted.get("scale"), "front scale"
    )


def build_front_options_context(
    *,
    call_capture: Mapping[str, object],
    put_capture: Mapping[str, object],
    prior_close: float,
    realised_volatility_20d: float,
) -> dict[str, float]:
    """Build the frozen D-1 ATM pair dimensions and soft-regime probabilities."""

    if (
        call_capture.get("source_completeness") != "complete"
        or put_capture.get("source_completeness") != "complete"
    ):
        raise ValueError("front option captures must be complete")
    if call_capture.get("option_right") != "call" or put_capture.get("option_right") != "put":
        raise ValueError("front option pair rights differ")
    if call_capture.get("expiry") != put_capture.get("expiry") or call_capture.get(
        "strike"
    ) != put_capture.get("strike"):
        raise ValueError("front option pair identity differs")
    call_bid = _required_number(call_capture.get("bid"), "call bid")
    call_ask = _required_number(call_capture.get("ask"), "call ask")
    put_bid = _required_number(put_capture.get("bid"), "put bid")
    put_ask = _required_number(put_capture.get("ask"), "put ask")
    call_iv = _required_number(call_capture.get("model_implied_volatility"), "call IV")
    put_iv = _required_number(put_capture.get("model_implied_volatility"), "put IV")
    if (
        min(call_bid, put_bid) < 0.0
        or call_ask < call_bid
        or put_ask < put_bid
        or min(call_iv, put_iv, prior_close, realised_volatility_20d) <= 0.0
    ):
        raise ValueError("front option pair values are invalid")
    call_mid = (call_bid + call_ask) / 2.0
    put_mid = (put_bid + put_ask) / 2.0
    total_mid = call_mid + put_mid
    if total_mid <= 0.0:
        raise ValueError("front option pair midpoint is unavailable")
    raw: dict[str, float | None] = {
        "atm_iv": (call_iv + put_iv) / 2.0,
        "straddle_mid_pct": total_mid / prior_close,
        "call_put_iv_gap": call_iv - put_iv,
        "skew_25d": None,
        "combined_relative_spread": ((call_ask - call_bid) + (put_ask - put_bid)) / total_mid,
        "iv_minus_realised_20d": (call_iv + put_iv) / 2.0 - realised_volatility_20d,
        "near_spot_oi_concentration": None,
        "call_put_oi_imbalance": None,
    }
    feature_data = _mapping(DATA["front_options_features"], "front feature data")
    medians = _mapping(feature_data["imputation_medians"], "front medians")
    scales = _mapping(feature_data["scales"], "front scales")
    imputed = {
        name: (_required_number(medians[name], f"{name} median") if value is None else value)
        for name, value in raw.items()
    }
    z_atm = _z(imputed["atm_iv"], _mapping(scales["atm_iv"], "atm scale"))
    z_straddle = _z(
        imputed["straddle_mid_pct"], _mapping(scales["straddle_mid_pct"], "straddle scale")
    )
    z_gap = _z(imputed["call_put_iv_gap"], _mapping(scales["call_put_iv_gap"], "gap scale"))
    z_skew = _z(imputed["skew_25d"], _mapping(scales["skew_25d"], "skew scale"))
    z_spread = _z(
        imputed["combined_relative_spread"],
        _mapping(scales["combined_relative_spread"], "spread scale"),
    )
    z_iv_rv = _z(
        imputed["iv_minus_realised_20d"],
        _mapping(scales["iv_minus_realised_20d"], "IV-RV scale"),
    )
    dimensions = {
        "front_options_implied_tension": (z_atm + z_straddle + z_iv_rv) / 3.0,
        "front_options_premium_richness": (z_straddle + z_iv_rv) / 2.0,
        "front_options_downside_asymmetry": (z_skew - z_gap) / 2.0,
        "front_options_liquidity_stress": z_spread,
        "front_options_positioning_concentration": _z(
            imputed["near_spot_oi_concentration"],
            _mapping(scales["near_spot_oi_concentration"], "OI concentration scale"),
        ),
        "front_options_directional_positioning": _z(
            imputed["call_put_oi_imbalance"],
            _mapping(scales["call_put_oi_imbalance"], "OI imbalance scale"),
        ),
        "front_options_surface_disagreement": (
            _z(
                abs(imputed["call_put_iv_gap"]),
                _mapping(scales["abs_call_put_iv_gap"], "absolute gap scale"),
            )
            + _z(
                abs(imputed["skew_25d"]),
                _mapping(scales["abs_skew_25d"], "absolute skew scale"),
            )
            + z_spread
        )
        / 3.0,
    }
    indicators = {
        "skew_25d_missing": 1.0,
        "near_spot_oi_concentration_missing": 1.0,
        "call_put_oi_imbalance_missing": 1.0,
    }
    regime = _mapping(DATA["front_options_regime"], "front regime data")
    inputs = {**dimensions, **indicators}
    values = [
        _required_number(inputs.get(str(name)), f"regime input {name}")
        for name in regime["input_columns"]
    ]
    log_density: list[float] = []
    for weight, means, variances in zip(
        regime["canonical_weights"],
        regime["canonical_input_means"],
        regime["canonical_covariances"],
        strict=True,
    ):
        variance_values = [max(float(value), 1e-12) for value in variances]
        log_density.append(
            math.log(float(weight))
            - 0.5
            * (
                len(values) * math.log(2.0 * math.pi)
                + math.fsum(math.log(value) for value in variance_values)
                + math.fsum(
                    (value - float(mean)) ** 2 / variance
                    for value, mean, variance in zip(values, means, variance_values, strict=True)
                )
            )
        )
    maximum = max(log_density)
    exponentials = [math.exp(value - maximum) for value in log_density]
    total = math.fsum(exponentials)
    probabilities = [value / total for value in exponentials]
    ordered = sorted(probabilities)
    result = {**dimensions, **indicators}
    result.update(
        {f"front_options_regime_p_{index}": value for index, value in enumerate(probabilities)}
    )
    result["front_options_regime_entropy"] = -math.fsum(
        value * math.log(max(value, 1e-15)) for value in probabilities
    )
    result["front_options_regime_margin"] = ordered[-1] - ordered[-2]
    if set(REQUIRED_GROUP_O_FEATURES).difference(result):
        raise ValueError("frozen Group-O construction is incomplete")
    return result


def _sum(values: Sequence[float]) -> float:
    return math.fsum(values) if all(math.isfinite(value) for value in values) else math.nan


def _mean(values: Sequence[float]) -> float:
    return _sum(values) / len(values) if values else math.nan


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return math.nan
    mean_x = (len(values) - 1) / 2.0
    mean_y = _mean(values)
    denominator = math.fsum((index - mean_x) ** 2 for index in range(len(values)))
    return (
        math.fsum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
        / denominator
    )


def _continuation_boundary(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> dict[str, float]:
    candidate: tuple[int, int, float, float] | None = None
    for position in range(max(6, len(highs) - 4), len(highs)):
        prior_high = max(highs[position - 6 : position])
        prior_low = min(lows[position - 6 : position])
        up = max(0.0, highs[position] - prior_high) / (abs(prior_high) + EPSILON)
        down = max(0.0, prior_low - lows[position]) / (abs(prior_low) + EPSILON)
        if up > 0.0 or down > 0.0:
            direction = 1 if up >= down else -1
            candidate = (
                position,
                direction,
                prior_high if direction > 0 else prior_low,
                max(up, down),
            )
    if candidate is None:
        return {
            "break_above": 0.0,
            "break_below": 0.0,
            "signed_distance": 0.0,
            "acceptance_count": 0.0,
            "rejection": 0.0,
        }
    position, direction, boundary, breach = candidate
    beyond = [
        value > boundary if direction > 0 else value < boundary for value in closes[position:]
    ]
    distance = (
        (closes[-1] - boundary) / (abs(boundary) + EPSILON)
        if direction > 0
        else (boundary - closes[-1]) / (abs(boundary) + EPSILON)
    )
    rejected = distance <= 0.0
    return {
        "break_above": float(direction > 0),
        "break_below": float(direction < 0),
        "signed_distance": direction * max(0.0, distance if not rejected else breach),
        "acceptance_count": float(direction * sum(beyond)),
        "rejection": float(-direction if rejected else 0.0),
    }


def _attempt_boundary(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    attempt_sign: int,
) -> dict[str, float]:
    marker = len(highs) - 1
    start = marker - 4
    prior_highs = highs[max(0, start - 6) : start]
    prior_lows = lows[max(0, start - 6) : start]
    attempt_highs = highs[start : marker - 1]
    attempt_lows = lows[start : marker - 1]
    response = closes[marker - 1 : marker + 1]
    if len(prior_highs) != 6 or len(attempt_highs) != 3 or len(response) != 2 or not attempt_sign:
        return {"failure": math.nan, "inside": math.nan, "maintained": math.nan}
    boundary = min(prior_lows) if attempt_sign < 0 else max(prior_highs)
    extreme = min(attempt_lows) if attempt_sign < 0 else max(attempt_highs)
    response_close = response[-1]
    failure = 0.0
    if attempt_sign < 0 and extreme < boundary and response_close > boundary:
        failure = (response_close - boundary) / (abs(boundary) + EPSILON)
    elif attempt_sign > 0 and extreme > boundary and response_close < boundary:
        failure = -(boundary - response_close) / (abs(boundary) + EPSILON)
    inside = (
        max(0.0, response_close - boundary) / (abs(boundary) + EPSILON)
        if attempt_sign < 0
        else -max(0.0, boundary - response_close) / (abs(boundary) + EPSILON)
    )
    maintained = (
        sum(value > boundary for value in response)
        if attempt_sign < 0
        else -sum(value < boundary for value in response)
    )
    return {"failure": failure, "inside": inside, "maintained": float(maintained)}


def build_direction_features(
    *,
    symbol: str,
    checkpoint: int,
    stock_prefix: Mapping[str, object],
    market_prefix: Mapping[str, object],
) -> dict[str, float]:
    """Reproduce frozen A1/C1/R1 T-1 raw fields from bounded prefix receipts."""

    if symbol not in COHORT or checkpoint not in CHECKPOINTS:
        raise ValueError("direction feature identity is outside the frozen grid")
    if (
        stock_prefix.get("bar_number") != checkpoint
        or market_prefix.get("bar_number") != checkpoint
    ):
        raise ValueError("direction prefixes do not match the checkpoint")
    stock_rows = {
        int(_required_number(row.get("bar_number"), "stock bar number")): row
        for row in (
            _mapping(value, "stock trailing bar")
            for value in _sequence(stock_prefix.get("trailing_bars"), "stock trailing bars")
        )
        if int(_required_number(row.get("bar_number"), "stock bar number")) < checkpoint
    }
    market_rows = {
        int(_required_number(row.get("bar_number"), "market bar number")): row
        for row in (
            _mapping(value, "market trailing bar")
            for value in _sequence(market_prefix.get("trailing_bars"), "market trailing bars")
        )
        if int(_required_number(row.get("bar_number"), "market bar number")) < checkpoint
    }
    numbers = tuple(sorted(set(stock_rows).intersection(market_rows)))[-11:]
    if (
        len(numbers) < 5
        or numbers[-1] != checkpoint - 1
        or numbers != tuple(range(numbers[0], checkpoint))
    ):
        raise ValueError("direction T-1 bar window is incomplete")
    rows = [stock_rows[number] for number in numbers]
    market = [market_rows[number] for number in numbers]
    returns = [
        math.log1p(_required_number(row.get("return_bps"), "stock return") / 10_000.0)
        for row in rows
    ]
    market_returns = [
        math.log1p(_required_number(row.get("return_bps"), "market return") / 10_000.0)
        for row in market
    ]
    relative = [stock - broad for stock, broad in zip(returns, market_returns, strict=True)]
    close = [_required_number(row.get("close"), "stock close") for row in rows]
    high = [_required_number(row.get("high"), "stock high") for row in rows]
    low = [_required_number(row.get("low"), "stock low") for row in rows]
    opening = [_required_number(row.get("open"), "stock open") for row in rows]
    activity = [
        _required_number(row.get("historical_relative_activity"), "stock activity") for row in rows
    ]
    vwap = [_required_number(row.get("session_vwap"), "session VWAP") for row in rows]
    width = [top - bottom for top, bottom in zip(high, low, strict=True)]
    clv = [
        (2.0 * final - top - bottom) / (spread + EPSILON)
        for final, top, bottom, spread in zip(close, high, low, width, strict=True)
    ]
    wick = [
        (min(start, final) - bottom - (top - max(start, final))) / (spread + EPSILON)
        for start, final, bottom, top, spread in zip(opening, close, low, high, width, strict=True)
    ]
    stock_1, stock_2, stock_4, stock_6 = (
        _sum(returns[-1:]),
        _sum(returns[-2:]),
        _sum(returns[-4:]),
        _sum(returns[-6:]),
    )
    relative_1, relative_2 = _sum(relative[-1:]), _sum(relative[-2:])
    net_sign = _sign(stock_4) if math.isfinite(stock_4) else 0
    continuation = _continuation_boundary(high, low, close)
    direction_closes = (
        [
            math.log(current / prior) * net_sign > 0.0
            for prior, current in zip(close[-5:-1], close[-4:], strict=True)
        ]
        if len(close) >= 5 and net_sign
        else [False] * 4
    )
    vwap_side = (
        [
            (final - value) * net_sign > 0.0
            for final, value in zip(close[-3:], vwap[-3:], strict=True)
        ]
        if net_sign
        else [False] * 3
    )
    attempt_returns = returns[-5:-2]
    response_returns = returns[-2:]
    response_market = market_returns[-2:]
    attempt_relative = relative[-5:-2]
    response_relative = relative[-2:]
    attempt_return = _sum(attempt_returns)
    response_return = _sum(response_returns)
    attempt_sign = _sign(attempt_return) if math.isfinite(attempt_return) else 0
    attempt_path = _sum([abs(value) for value in attempt_returns])
    attempt_efficiency = abs(attempt_return) / (attempt_path + EPSILON)
    response_efficiency = (
        attempt_sign
        * response_return
        / (_sum([abs(value) for value in response_returns]) + EPSILON)
        if attempt_sign
        else 0.0
    )
    response_activity = _mean(activity[-2:])
    attempt_activity = _mean(activity[-5:-2])
    attempt_impact = abs(attempt_return) / (attempt_activity + EPSILON)
    response_impact = abs(response_return) / (response_activity + EPSILON)
    attempted_response_impact = max(0.0, attempt_sign * response_return) / (
        response_activity + EPSILON
    )
    boundary = _attempt_boundary(high, low, close, attempt_sign)
    response_clv = _mean(clv[-2:])
    response_wick = _mean(wick[-2:])
    response_close_failure = (
        -attempt_sign * (1.0 - attempt_sign * response_clv) / 2.0 if attempt_sign else 0.0
    )
    vwap_reclaim = 0.0
    if attempt_sign < 0 and close[-3] < vwap[-3] and close[-1] > vwap[-1]:
        vwap_reclaim = 1.0
    elif attempt_sign > 0 and close[-3] > vwap[-3] and close[-1] < vwap[-1]:
        vwap_reclaim = -1.0
    marker_vwap_distance = math.log(close[-1] / vwap[-1])
    raw = {
        "c_z_return_5m": stock_1,
        "c_z_return_10m": stock_2,
        "c_z_return_20m": stock_4,
        "c_z_return_30m": stock_6,
        "c_directional_efficiency_20m": _sum(returns[-4:])
        / (_sum([abs(value) for value in returns[-4:]]) + EPSILON),
        "c_mean_clv_4": _mean(clv[-4:]),
        "c_directional_close_fraction_4": net_sign
        * _mean([float(value) for value in direction_closes]),
        "c_signed_wick_asymmetry_4": _mean(wick[-4:]),
        "c_signed_vwap_slope_4": _slope([math.log(value) for value in vwap[-4:]]),
        "c_signed_vwap_distance": marker_vwap_distance,
        "c_vwap_side_closes_3": float(net_sign * sum(vwap_side)),
        "c_break_above_prior_six_high": continuation["break_above"],
        "c_break_below_prior_six_low": continuation["break_below"],
        "c_signed_boundary_distance": continuation["signed_distance"],
        "c_signed_boundary_acceptance_count": continuation["acceptance_count"],
        "c_boundary_rejection": continuation["rejection"],
        "c_relative_return_5m": relative_1,
        "c_relative_return_10m": relative_2,
        "c_relative_agreement": (
            _sign(stock_2) * abs(relative_2) if _sign(stock_2) == _sign(relative_2) else 0.0
        ),
        "a_attempt_return_abs": abs(attempt_return),
        "a_attempt_path_length": attempt_path,
        "a_attempt_directional_efficiency": attempt_efficiency,
        "a_response_followthrough": response_return,
        "a_reversal_efficiency_change": -attempt_sign * (attempt_efficiency - response_efficiency),
        "a_wick_rejection": response_wick,
        "a_close_location_recovery": response_clv,
        "a_failure_close_near_extreme": response_close_failure,
        "a_boundary_failure": boundary["failure"],
        "a_boundary_distance_inside": boundary["inside"],
        "a_boundary_maintenance_count": boundary["maintained"],
        "a_vwap_reclaim_failure": vwap_reclaim,
        "a_vwap_distance_after_failure": marker_vwap_distance if vwap_reclaim else 0.0,
        "a_attempt_price_impact": attempt_impact,
        "a_response_price_impact": response_impact,
        "a_price_impact_decline": -attempt_sign * (attempt_impact - attempted_response_impact),
        "a_elevated_activity_weak_progress": -attempt_sign
        * response_activity
        * max(0.0, attempt_efficiency - response_efficiency),
        "a_relative_recovery": _sum(response_relative) - _sum(attempt_relative),
        "a_market_resilience": (
            -attempt_sign
            * max(0.0, attempt_sign * _sum(response_market))
            * max(0.0, -attempt_sign * _sum(response_relative))
            if attempt_sign
            else 0.0
        ),
    }
    group = "early" if checkpoint <= 14 else "middle" if checkpoint <= 24 else "late"
    beta_rows = cast(Sequence[Mapping[str, str]], DATA["direction_beta"])
    beta = next(
        (row for row in beta_rows if row["stock"] == symbol and row["checkpoint_group"] == group),
        None,
    )
    if beta is None:
        raise ValueError("frozen direction beta is unavailable")
    stock_lags = [returns[-1 - lag] for lag in range(4)]
    market_lags = [market_returns[-1 - lag] for lag in range(4)]
    alpha, beta_value = float(beta["alpha"]), float(beta["beta"])
    residuals = [
        stock_value - (alpha + beta_value * market_value)
        for stock_value, market_value in zip(stock_lags, market_lags, strict=True)
    ]
    residual_5, residual_10, residual_20 = (
        _sum(residuals[:1]),
        _sum(residuals[:2]),
        _sum(residuals[:4]),
    )
    stock_20, market_20 = _sum(stock_lags), _sum(market_lags)
    residual_slope = _slope(tuple(reversed(residuals)))
    low_range = float(beta["residual_range_low"]) * math.sqrt(4.0)
    high_range = float(beta["residual_range_high"]) * math.sqrt(4.0)
    distance = (
        residual_20 - high_range
        if residual_20 > high_range
        else residual_20 - low_range
        if residual_20 < low_range
        else 0.0
    )
    raw.update(
        {
            "r_residual_return_5m": residual_5,
            "r_residual_return_10m": residual_10,
            "r_residual_return_20m": residual_20,
            "r_residual_slope": residual_slope,
            "r_residual_persistence": _mean([float(_sign(value)) for value in residuals]),
            "r_change_in_residual_strength": _mean(residuals[:2]) - _mean(residuals[2:]),
            "r_stock_flat_up_market_down": abs(market_20)
            if stock_20 >= 0.0 and market_20 < 0.0
            else 0.0,
            "r_stock_flat_down_market_up": -abs(market_20)
            if stock_20 <= 0.0 and market_20 > 0.0
            else 0.0,
            "r_residual_volatility_score": residual_20
            / (float(beta["residual_scale"]) * math.sqrt(4.0) + EPSILON),
            "r_distance_from_normal_residual_range": distance,
            "r_absolute_residual_direction_agreement": _sign(residual_20) * abs(residual_20)
            if _sign(stock_20) == _sign(residual_20)
            else 0.0,
            "r_improving_while_absolute_compressed": residual_slope
            if abs(stock_20) <= 4.0 * float(beta["stock_abs_return_median"])
            else 0.0,
        }
    )
    if len(raw) != 50:
        raise ValueError("frozen direction feature construction is incomplete")
    return raw


_NORMALISATION_ROWS = cast(
    Sequence[Mapping[str, Any]],
    cast(Mapping[str, Any], DATA["direction_normalisation"])["parameters"],
)
_NORMALISATION_EXACT = {
    (str(item["feature"]), str(item["stock"]), int(item["checkpoint"])): item
    for item in _NORMALISATION_ROWS
    if item["stock"] != "__POOLED__"
}
_NORMALISATION_POOLED = {
    str(item["feature"]): item for item in _NORMALISATION_ROWS if item["stock"] == "__POOLED__"
}


def classify_directions(
    *, symbol: str, checkpoint: int, session: str, raw_features: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    """Apply the three separate frozen A1, C1, and R1 classifiers without fitting."""

    models = _mapping(DATA["direction_models"], "direction models")
    thresholds = _mapping(DATA["direction_thresholds"], "direction thresholds")
    categories = {
        "stock": symbol,
        "checkpoint_category": str(checkpoint),
        "day_of_week": date.fromisoformat(session).strftime("%A"),
    }
    results: list[dict[str, object]] = []
    for model_id in ("A1", "C1", "R1"):
        model = _mapping(models[model_id], f"{model_id} model")
        numeric = tuple(str(value) for value in model["numeric_features"])
        normalised: dict[str, float] = {}
        fallback: dict[str, str] = {}
        for name in numeric:
            fitted = _NORMALISATION_EXACT.get(
                (name, symbol, checkpoint), _NORMALISATION_POOLED[name]
            )
            raw = _finite(raw_features.get(name))
            value = float(fitted["missing_value"]) if raw is None else raw
            clipped = min(max(value, float(fitted["clip_lower"])), float(fitted["clip_upper"]))
            normalised[name] = (clipped - float(fitted["median"])) / float(fitted["iqr"])
            fallback[name] = str(fitted["fallback_level"])
        design: list[float] = []
        for name in numeric:
            value = normalised[name]
            missing = not math.isfinite(value)
            if missing:
                value = float(model["medians"][name])
            design.extend(
                (
                    (value - float(model["robust_centers"][name]))
                    / float(model["robust_scales"][name]),
                    float(missing),
                )
            )
        for category in model["categorical_features"]:
            name = str(category)
            levels = tuple(str(value) for value in model["categorical_levels"][name])
            observed = categories[name] if categories[name] in levels else "__UNKNOWN__"
            design.extend(float(observed == level) for level in levels)
        coefficients = tuple(float(value) for value in model["coefficients"])
        if len(design) != len(coefficients):
            raise ValueError("frozen direction design width differs")
        probability = _sigmoid(
            math.fsum(
                value * coefficient for value, coefficient in zip(design, coefficients, strict=True)
            )
            + float(model["intercept"])
        )
        boundary = float(_mapping(thresholds[model_id], "direction threshold")["boundary"])
        action = "ABSTAIN"
        if probability >= 0.5 + boundary:
            action = "CALL"
        elif probability <= 0.5 - boundary:
            action = "PUT"
        results.append(
            {
                "model_id": model_id,
                "probability_up": probability,
                "confidence": abs(probability - 0.5),
                "action": action,
                "boundary": boundary,
                "label": _LABELS[model_id],
                "model_hash": ARTIFACT_HASHES["direction_models"],
                "preprocessing_hash": ARTIFACT_HASHES["direction_normalisation"],
                "feature_hash": _hash_json(normalised),
                "fallback_levels": tuple(sorted(set(fallback.values()))),
            }
        )
    return tuple(results)


__all__ = [
    "ARTIFACT_HASHES",
    "CAUSAL_GROUP_I_FEATURES",
    "CHECKPOINTS",
    "COHORT",
    "M1C_THRESHOLD",
    "MINIMUM_EPISODE_SPACING_MINUTES",
    "REQUIRED_GROUP_O_FEATURES",
    "build_causal_group_i",
    "build_direction_features",
    "build_front_options_context",
    "build_group_i_for_symbol",
    "classify_directions",
    "score_m1c",
]
