"""T-218 — weather in two roles, kept apart (FR-206, FR-751a, AC-18).

The whole risk here is that one endpoint serves both roles and the response
does not say which one you asked for. A prior and a leak differ only by a date
range. So the tests below are less about weather than about that boundary:
that the prior provably cannot see the race, and that the firewall catches a
feature which does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.data.weather import (
    ARCHIVE_START_YEAR,
    MIN_PRIOR_YEARS,
    PRIOR_WINDOW_DAYS,
    WET_DAY_MM,
    WeatherError,
    attach,
    climatological_prior,
    collect_archive,
    provenance,
    race_observation,
    race_weather_table,
)
from f1pred.features.build import build_oracle, build_primary
from f1pred.features.checks import (
    POST_RACE_COLUMNS,
    behavioural_leakage_check,
    static_leakage_check,
)
from f1pred.features.registry import (
    REGISTRY,
    Availability,
    Feature,
    FeatureGroup,
)
from tests.fixtures.races import synthetic_corpus

RACE_DATE = pd.Timestamp("2015-03-15")
CIRCUIT = "albert_park"


def make_archive(
    *, years=range(2000, 2026), wet_share: float = 0.5, race_day_precip: float = 0.0
) -> pd.DataFrame:
    """A synthetic daily archive with a controlled wet-day rate."""
    frames = []
    for year in years:
        dates = pd.date_range(f"{year}-02-03", f"{year}-04-24", freq="D")
        offsets = np.arange(len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "circuit_id": CIRCUIT,
                    "date": dates,
                    "precipitation_sum": np.where(
                        offsets % 2 == 0, WET_DAY_MM + 2.0 if wet_share else 0.0, 0.0
                    ),
                    "temperature_2m_max": 20.0 + offsets * 0.01,
                    "temperature_2m_mean": 15.0,
                    "wind_speed_10m_max": 10.0,
                }
            )
        )
    rows = frames
    df = pd.concat(rows, ignore_index=True)
    day = (df["circuit_id"] == CIRCUIT) & (df["date"] == RACE_DATE)
    df.loc[day, "precipitation_sum"] = race_day_precip
    return df


class TestPriorCannotSeeTheRace:
    """FR-206. The single most important property in this module."""

    def test_the_race_day_itself_does_not_move_the_prior(self):
        dry = climatological_prior(make_archive(race_day_precip=0.0), CIRCUIT, RACE_DATE)
        soaked = climatological_prior(
            make_archive(race_day_precip=999.0), CIRCUIT, RACE_DATE
        )
        assert dry == soaked

    def test_the_races_whole_season_does_not_move_the_prior(self):
        archive = make_archive()
        tampered = archive.copy()
        same_year = tampered["date"].dt.year == RACE_DATE.year
        tampered.loc[same_year, "precipitation_sum"] = 500.0
        assert climatological_prior(archive, CIRCUIT, RACE_DATE) == climatological_prior(
            tampered, CIRCUIT, RACE_DATE
        )

    def test_an_earlier_year_does_move_the_prior(self):
        # The mirror of the two above: if nothing moved it, it would be a
        # constant and the tests would pass for the wrong reason.
        archive = make_archive()
        tampered = archive.copy()
        earlier = tampered["date"].dt.year == RACE_DATE.year - 3
        tampered.loc[earlier, "precipitation_sum"] = 500.0
        assert climatological_prior(archive, CIRCUIT, RACE_DATE) != climatological_prior(
            tampered, CIRCUIT, RACE_DATE
        )

    def test_a_race_with_no_earlier_years_gets_nulls_not_a_default(self):
        archive = make_archive(years=range(2015, 2020))
        got = climatological_prior(archive, CIRCUIT, RACE_DATE)
        assert np.isnan(got["weather_prior_wet_probability"])
        assert got["weather_prior_years"] == 0.0

    def test_too_few_years_yields_nulls_and_reports_the_count(self):
        archive = make_archive(years=range(2012, 2016))
        got = climatological_prior(archive, CIRCUIT, RACE_DATE, min_years=MIN_PRIOR_YEARS)
        assert np.isnan(got["weather_prior_wet_probability"])
        assert 0 < got["weather_prior_years"] < MIN_PRIOR_YEARS

    def test_an_unknown_circuit_yields_nulls(self):
        got = climatological_prior(make_archive(), "no_such_circuit", RACE_DATE)
        assert np.isnan(got["weather_prior_temp_max_c"])


class TestWindow:
    def test_the_window_wraps_around_new_year(self):
        # A December race must draw on early January too, or half its window
        # silently disappears.
        archive = make_archive()
        archive["date"] = archive["date"] + pd.to_timedelta(290, unit="D")
        december = pd.Timestamp("2015-12-30")
        got = climatological_prior(archive, CIRCUIT, december)
        assert got["weather_prior_years"] > 0

    def test_the_window_is_declared_not_magic(self):
        assert PRIOR_WINDOW_DAYS > 0 and ARCHIVE_START_YEAR >= 1990


class TestObservation:
    def test_it_returns_the_race_day(self):
        archive = make_archive(race_day_precip=12.0)
        assert race_observation(archive, CIRCUIT, RACE_DATE)[
            "weather_actual_precip_mm"
        ] == 12.0

    def test_a_missing_day_yields_nulls(self):
        got = race_observation(make_archive(), CIRCUIT, pd.Timestamp("1975-01-01"))
        assert np.isnan(got["weather_actual_precip_mm"])


class TestAttach:
    @pytest.fixture
    def corpus(self):
        return synthetic_corpus(seasons=(2018, 2019), rounds=3)

    def test_row_count_is_preserved(self, corpus):
        weather = race_weather_table(make_archive(), corpus)
        assert len(attach(corpus, weather)) == len(corpus)

    def test_a_duplicated_race_key_is_caught(self, corpus):
        weather = race_weather_table(make_archive(), corpus)
        with pytest.raises(WeatherError, match="duplicate"):
            attach(corpus, pd.concat([weather, weather], ignore_index=True))

    def test_every_race_gets_a_row(self, corpus):
        weather = race_weather_table(make_archive(), corpus)
        assert len(weather) == corpus[["season", "round"]].drop_duplicates().shape[0]


class TestRegistration:
    """FR-751a: two roles, and the registry keeps them apart."""

    def test_the_prior_features_are_pre_race(self):
        for name in ("weather_prior_wet", "weather_prior_temp", "weather_prior_wind"):
            f = REGISTRY.get(name)
            assert f.availability is Availability.PRE_RACE
            assert f.never_impute, "a thin archive is an absence, not a zero"

    def test_the_observed_features_are_post_race(self):
        for name in ("weather_actual_precip", "weather_actual_temp"):
            assert REGISTRY.get(name).availability is Availability.POST_RACE

    def test_observed_weather_never_reaches_the_primary_set(self):
        primary = {f.name for f in REGISTRY.primary()}
        assert "weather_actual_precip" not in primary
        assert "weather_actual_temp" not in primary

    def test_observed_weather_is_reachable_from_the_oracle_set(self):
        oracle = {f.name for f in REGISTRY.oracle()}
        assert "weather_actual_precip" in oracle

    def test_observed_weather_is_perturbed_by_the_behavioural_check(self):
        # Without this the negative control below would pass both layers.
        assert "weather_actual_precip_mm" in POST_RACE_COLUMNS


class TestNegativeControl:
    """A weather feature that peeks at race day must be caught."""

    @pytest.fixture
    def corpus(self):
        base = synthetic_corpus(seasons=(2018, 2019), rounds=4)
        return attach(base, race_weather_table(make_archive(), base))

    def _leaky(self):
        reg = REGISTRY.fork("leaky-weather")

        def weather_leak(df: pd.DataFrame) -> pd.Series:
            """Reads the rain that fell during the race it is predicting."""
            return pd.to_numeric(df["weather_actual_precip_mm"], errors="coerce")

        reg.register(
            Feature(
                name="weather_leak",
                fn=weather_leak,
                availability=Availability.PRE_RACE,
                group=FeatureGroup.PRIMARY,
                description="Reads race-day rainfall.",
                depends_on=("weather_actual_precip_mm",),
            )
        )
        return reg

    def test_the_behavioural_check_catches_it(self, corpus):
        reg = self._leaky()
        report = behavioural_leakage_check(
            corpus, features=reg.primary(), registry=reg, races=[(2019, 2)]
        )
        assert not report.passed
        assert any("weather_leak" in v for v in report.violations)

    def test_the_static_check_alone_does_not(self, corpus):
        # It declares only a raw column, so nothing in the declarations is
        # wrong. This is why FR-206 needs the behavioural layer, and why the
        # observed-weather columns had to join the perturbation set.
        assert static_leakage_check(self._leaky().primary(), registry=self._leaky()).passed

    def test_the_honest_prior_feature_passes_the_same_check(self, corpus):
        report = behavioural_leakage_check(corpus, count=3)
        report.raise_if_failed()


class TestProvenance:
    def test_the_record_names_the_source_licence_and_split(self):
        got = provenance("2026-08-31")
        assert got["retrieved"] == "2026-08-31"
        for field in ("source", "url", "licence", "role_split"):
            assert got[field]
        assert "PRE_RACE" in got["role_split"] and "POST_RACE" in got["role_split"]


class TestCollection:
    def test_a_failing_circuit_is_recorded_not_fatal(self):
        class Client:
            def circuit_daily(self, circuit_id, *, start_year, end_year):
                if circuit_id == "monaco":
                    raise WeatherError("boom")
                return make_archive().assign(circuit_id=circuit_id)

        archive, failed = collect_archive(
            ["albert_park", "monaco"], client=Client(), log=lambda _: None
        )
        assert failed == ["monaco"]
        assert set(archive["circuit_id"]) == {"albert_park"}
