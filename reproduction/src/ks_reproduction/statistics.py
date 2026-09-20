"""Numerical summaries and named aggregate-law regressions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from .config import ALMCoefficients


@dataclass(frozen=True)
class SummaryStatistics:
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float

    @classmethod
    def from_array(cls, values: np.ndarray) -> SummaryStatistics:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("cannot summarize an empty/non-finite sample")
        return cls(
            count=int(finite.size),
            mean=float(finite.mean()),
            standard_deviation=float(finite.std(ddof=0)),
            minimum=float(finite.min()),
            maximum=float(finite.max()),
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class OnlineMoments:
    """Mergeable population moments for memory-bounded evaluation."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        other_count = int(values.size)
        other_mean = float(values.mean())
        other_m2 = float(np.square(values - other_mean).sum())
        if self.count == 0:
            self.count, self.mean, self.m2 = other_count, other_mean, other_m2
        else:
            delta = other_mean - self.mean
            total = self.count + other_count
            self.mean += delta * other_count / total
            self.m2 += other_m2 + delta * delta * self.count * other_count / total
            self.count = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def summary(self) -> SummaryStatistics:
        if self.count == 0:
            raise ValueError("no finite observations accumulated")
        return SummaryStatistics(
            count=self.count,
            mean=self.mean,
            standard_deviation=math.sqrt(self.m2 / self.count),
            minimum=self.minimum,
            maximum=self.maximum,
        )


@dataclass(frozen=True)
class TTest:
    statistic: float
    degrees_of_freedom: float
    p_value_two_sided: float


def independent_t_test(
    first: SummaryStatistics,
    second: SummaryStatistics,
    *,
    equal_variance: bool = True,
) -> TTest:
    """Two-sample t test using sample variances reconstructed from summaries."""

    if first.count < 2 or second.count < 2:
        raise ValueError("a two-sample t test needs at least two observations per group")
    n1, n2 = first.count, second.count
    # Stored standard deviations use ddof=0, matching the thesis summary. Convert
    # to unbiased sample variances for the test itself.
    v1 = first.standard_deviation**2 * n1 / (n1 - 1)
    v2 = second.standard_deviation**2 * n2 / (n2 - 1)
    if equal_variance:
        degrees = float(n1 + n2 - 2)
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / degrees
        standard_error = math.sqrt(pooled * (1.0 / n1 + 1.0 / n2))
    else:
        first_term, second_term = v1 / n1, v2 / n2
        standard_error = math.sqrt(first_term + second_term)
        degrees = (first_term + second_term) ** 2 / (
            first_term**2 / (n1 - 1) + second_term**2 / (n2 - 1)
        )
    statistic = (first.mean - second.mean) / standard_error
    # At the thesis sample size (millions), the normal tail and Student tail
    # agree far beyond the printed precision. scipy is deliberately unnecessary.
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return TTest(statistic=statistic, degrees_of_freedom=degrees, p_value_two_sided=p_value)


@dataclass(frozen=True)
class RegimeFit:
    intercept: float
    slope: float
    r_squared: float
    rmse: float
    observations: int


@dataclass(frozen=True)
class ALMRegression:
    coefficients: ALMCoefficients
    bad: RegimeFit
    good: RegimeFit

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": {
                "bad_intercept": self.coefficients.bad_intercept,
                "bad_slope": self.coefficients.bad_slope,
                "good_intercept": self.coefficients.good_intercept,
                "good_slope": self.coefficients.good_slope,
            },
            "bad": asdict(self.bad),
            "good": asdict(self.good),
        }


def _regime_fit(current: np.ndarray, following: np.ndarray) -> RegimeFit:
    valid = (
        np.isfinite(current)
        & np.isfinite(following)
        & (current > 0.0)
        & (following > 0.0)
    )
    x, y = np.log(current[valid]), np.log(following[valid])
    if x.size < 2:
        raise ValueError("each ALM regime needs at least two positive observations")
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction_log = intercept + slope * x
    residual = y - prediction_log
    denominator = np.square(y - y.mean()).sum()
    r_squared = 1.0 - float(np.square(residual).sum() / denominator) if denominator > 0 else 1.0
    # The thesis reports RMSE in capital levels, not log units.
    prediction_level = np.exp(prediction_log)
    rmse = float(np.sqrt(np.mean(np.square(following[valid] - prediction_level))))
    return RegimeFit(
        intercept=float(intercept),
        slope=float(slope),
        r_squared=r_squared,
        rmse=rmse,
        observations=int(x.size),
    )


def regress_alm(
    capital_paths: np.ndarray | Iterable[np.ndarray],
    aggregate_state_paths: np.ndarray | Iterable[np.ndarray],
) -> ALMRegression:
    capital_values = (
        list(capital_paths)
        if not isinstance(capital_paths, np.ndarray)
        else capital_paths
    )
    capital = np.asarray(capital_values)
    states = np.asarray(
        list(aggregate_state_paths)
        if not isinstance(aggregate_state_paths, np.ndarray)
        else aggregate_state_paths
    )
    if capital.ndim == 1:
        capital = capital[None, :]
    if states.ndim == 1:
        states = states[None, :]
    if capital.shape != states.shape or capital.shape[1] < 2:
        raise ValueError("capital and aggregate-state paths need matching [paths, time] shapes")
    current = capital[:, :-1].reshape(-1)
    following = capital[:, 1:].reshape(-1)
    regime = states[:, :-1].reshape(-1)
    bad = _regime_fit(current[regime == 0], following[regime == 0])
    good = _regime_fit(current[regime == 1], following[regime == 1])
    coefficients = ALMCoefficients(
        bad_intercept=bad.intercept,
        bad_slope=bad.slope,
        good_intercept=good.intercept,
        good_slope=good.slope,
    )
    return ALMRegression(coefficients=coefficients, bad=bad, good=good)
