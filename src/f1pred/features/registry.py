"""The feature registry (FR-201, FR-202, FR-207, FR-209, FR-213). Plan D-1.

A feature is not a column name. It is a declaration — of when its inputs become
knowable, of which group it belongs to, of whether it expresses itself relative
to the contemporaneous field — bound to the function that computes it. The
primary feature set is *derived* from those declarations and is never written
by hand.

v0 had the hand-written list. `weather_mixed` sat in it for four months,
computed from the race that had already happened, and nothing in the codebase
was capable of noticing. The registry exists so that the same mistake becomes
a registration error rather than an inflated score.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

import pandas as pd

from f1pred.data.schema import RACE_ROW_SCHEMA

#: What a feature function receives and returns. It is handed a frame carrying
#: the canonical race rows plus every feature it declared in `depends_on`, and
#: returns one Series aligned to that frame's index.
FeatureFn = Callable[[pd.DataFrame], pd.Series]


class RegistryError(ValueError):
    """Raised when a feature declaration is malformed or contradictory."""


class Availability(Enum):
    """When a feature's inputs become knowable (FR-201).

    The distinction is about the *race being predicted*, not about the data in
    general. A rolling average of a driver's past finishes reads finishing
    positions — a post-race quantity — but only from races that had already
    happened, so it is PRE_RACE. Whether that is actually true of the
    implementation is what FR-205's behavioural check decides; this field only
    records the claim.
    """

    PRE_RACE = "pre_race"
    POST_RACE = "post_race"


class FeatureGroup(Enum):
    """A named partition of the registry, orthogonal to availability.

    FR-213 requires the telemetry features to be a group distinct from the
    primary set, because FR-214 leaves them undefined for roughly half the
    corpus and FR-215 therefore measures them as a separate arm.
    """

    PRIMARY = "primary"
    TELEMETRY = "telemetry"
    JUNIOR_FORMULA = "junior_formula"
    #: A stepping stone: computed, available as a dependency, never itself a
    #: model input. FR-209 requires historical-performance features to be
    #: field-relative, and the natural way to build a percentile is to compute
    #: the absolute quantity first. Without this group that absolute quantity
    #: would have to either enter the primary set — exactly what FR-209
    #: forbids — or be recomputed inside every feature that needs it.
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True, slots=True)
class Feature:
    """One registered feature and everything declared about it."""

    name: str
    fn: FeatureFn
    availability: Availability
    group: FeatureGroup
    description: str
    #: Raw schema columns and/or other feature names this function reads.
    #: `build` resolves these into a computation order, so a wrong or missing
    #: entry surfaces as a KeyError at build time rather than as a silent
    #: dependence on whatever happened to be computed first.
    depends_on: tuple[str, ...] = ()
    #: FR-209: does this summarise historical performance (pace, form,
    #: reliability)? If so it must be relative, or say why it cannot be.
    summarises_history: bool = False
    #: FR-209: expressed as a rank, percentile, or within-season z-score
    #: against the contemporaneous field, so its meaning survives a
    #: regulation reset.
    relative: bool = False
    #: Required when `summarises_history` and not `relative`.
    relative_justification: str | None = None
    #: FR-211: a null here is a fact, not a gap. The cold-start fallbacks of
    #: FR-212 skip such a feature — filling a sprint result on a weekend that
    #: had no sprint would assert an event that did not happen, which is the
    #: specific thing FR-211 forbids.
    never_impute: bool = False

    @property
    def is_primary(self) -> bool:
        """FR-202: admissible in the primary matrix."""
        return (
            self.availability is Availability.PRE_RACE
            and self.group is FeatureGroup.PRIMARY
        )


class Registry:
    """An ordered collection of feature declarations.

    Registration order is preserved and is the only ordering the build uses as
    a tie-break, so a rebuild produces byte-identical column order (FR-208).

    Tests fork the default registry rather than mutating it: the negative
    control of T-207 needs a deliberately leaky feature present for the checks
    to catch, and a leaky feature must never survive into a real build.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._features: dict[str, Feature] = {}

    # -- registration -------------------------------------------------------

    def register(self, feature: Feature) -> Feature:
        _validate(feature)
        if feature.name in self._features:
            raise RegistryError(
                f"feature {feature.name!r} is already registered in "
                f"{self.name!r}. Two features cannot share a name: the build "
                f"resolves dependencies by name and would silently pick one."
            )
        self._features[feature.name] = feature
        return feature

    def fork(self, name: str | None = None) -> Registry:
        """A copy that can be mutated without touching the default registry."""
        clone = Registry(name or f"{self.name}-fork")
        clone._features = dict(self._features)
        return clone

    # -- lookup -------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._features

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features.values())

    def get(self, name: str) -> Feature:
        try:
            return self._features[name]
        except KeyError:
            raise RegistryError(
                f"no feature named {name!r} in registry {self.name!r}. "
                f"FR-202 forbids building an unregistered feature — declare it "
                f"with @feature rather than reaching for the column directly."
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._features)

    # -- derived sets -------------------------------------------------------

    def select(
        self,
        *,
        availability: Availability | None = None,
        group: FeatureGroup | None = None,
    ) -> tuple[Feature, ...]:
        return tuple(
            f
            for f in self
            if (availability is None or f.availability is availability)
            and (group is None or f.group is group)
        )

    def primary(self) -> tuple[Feature, ...]:
        """FR-202: the primary feature set, derived, never hand-written."""
        return tuple(f for f in self if f.is_primary)

    def oracle(self) -> tuple[Feature, ...]:
        """FR-207: primary plus post-race, for upper-bound analysis only.

        Deliberately a separate method rather than a flag on `primary()`. A
        default argument that widens the feature set is one keyword away from
        being passed by accident; a differently named method is not.
        """
        return tuple(f for f in self if f.group is FeatureGroup.PRIMARY)

    def group_features(self, group: FeatureGroup) -> tuple[Feature, ...]:
        return self.select(group=group)


def _validate(feature: Feature) -> None:
    if not feature.name or not feature.name.strip():
        raise RegistryError("a feature must have a non-empty name")
    if feature.name in RACE_ROW_SCHEMA:
        raise RegistryError(
            f"feature {feature.name!r} collides with a raw schema column. "
            f"`depends_on` resolves names against raw columns and features "
            f"alike, so a shared name makes a dependency ambiguous — and a "
            f"feature that shadows its own input is a cycle."
        )
    if not feature.description.strip():
        raise RegistryError(
            f"feature {feature.name!r} has no description. Give the function a "
            f"docstring — an undocumented feature cannot be argued with."
        )

    # FR-209, enforced at declaration time rather than only in the static
    # check, so a non-relative pace feature cannot reach a model even
    # transiently. Intermediates are exempt because they are by definition not
    # model inputs; what they feed is checked instead.
    if feature.summarises_history and feature.group is not FeatureGroup.INTERMEDIATE:
        if not feature.relative and not (feature.relative_justification or "").strip():
            raise RegistryError(
                f"feature {feature.name!r} summarises historical performance "
                f"but is neither relative nor justified. FR-209: express it "
                f"against the contemporaneous field, or declare in "
                f"`relative_justification` why it cannot be. An absolute pace "
                f"number does not survive a regulation reset."
            )
    elif feature.relative_justification:
        raise RegistryError(
            f"feature {feature.name!r} carries a relative_justification but "
            f"does not declare summarises_history. The justification would "
            f"never be read; either declare the feature historical or drop "
            f"the justification."
        )

    if feature.relative and feature.relative_justification:
        raise RegistryError(
            f"feature {feature.name!r} is declared relative *and* carries a "
            f"justification for not being relative. Only one can be true."
        )


#: The registry the real features register into. `definitions` populates it on
#: import; `build` and `checks` read it.
REGISTRY: Final[Registry] = Registry()


def feature(
    availability: Availability,
    *,
    depends_on: Sequence[str] = (),
    group: FeatureGroup = FeatureGroup.PRIMARY,
    summarises_history: bool = False,
    relative: bool = False,
    relative_justification: str | None = None,
    never_impute: bool = False,
    name: str | None = None,
    registry: Registry | None = None,
) -> Callable[[FeatureFn], FeatureFn]:
    """Register a feature function. The decorated function is returned as-is.

    The feature's name defaults to the function's name and its description to
    the first line of its docstring, so the declaration and the documentation
    cannot drift apart.
    """

    def decorate(fn: FeatureFn) -> FeatureFn:
        doc = (fn.__doc__ or "").strip()
        target = registry if registry is not None else REGISTRY
        target.register(
            Feature(
                name=name or fn.__name__,
                fn=fn,
                availability=availability,
                group=group,
                description=doc.splitlines()[0] if doc else "",
                depends_on=tuple(depends_on),
                summarises_history=summarises_history,
                relative=relative,
                relative_justification=relative_justification,
                never_impute=never_impute,
            )
        )
        return fn

    return decorate


__all__ = [
    "Availability",
    "Feature",
    "FeatureFn",
    "FeatureGroup",
    "REGISTRY",
    "Registry",
    "RegistryError",
    "feature",
]
