"""
Attack-spec contract tests.

These lock two invariants that were previously enforced only by luck:

  1. Every shipped spec loads and validates.
  2. Every spec's amount band clears the Plausibility Gate's economic floor for
     the acquisition vector it claims.

Invariant 2 matters because the corpus builder draws amounts uniformly from the
band and asserts that the gate accepts every generated payload. If the band dips
below the floor, the builder crashes only on the seeds whose draws happen to land
low -- a flaky failure that looks like a generator bug but is really a spec bug.
ATTACK_6 shipped in exactly that state (band opened at $180 while claiming a
$200 synthetic-identity kit) and crashed the family-coverage run mid-sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pydantic
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from data.corpus_builder import SPEC_FILES, _SYNTHESIZERS  # noqa: E402
from schemas.attack import OperationalConstraints, load_attack_spec  # noqa: E402
from schemas.payment import RESOURCE_COST_TABLE_USD  # noqa: E402

SPEC_PATHS = sorted((BACKEND_ROOT / "attack_specs").glob("*.yaml"))


def test_spec_files_discovered():
    """Guard against an empty glob silently passing every parametrised test."""
    assert len(SPEC_PATHS) >= 8, f"expected >=8 specs, found {len(SPEC_PATHS)}"


@pytest.mark.parametrize("path", SPEC_PATHS, ids=lambda p: p.stem)
def test_spec_validates(path: Path):
    spec = load_attack_spec(path)
    assert spec.spec_id
    assert spec.taxon


@pytest.mark.parametrize("path", SPEC_PATHS, ids=lambda p: p.stem)
def test_amount_band_clears_economic_floor(path: Path):
    """The whole band -- not just its midpoint -- must be gate-satisfiable."""
    cons = load_attack_spec(path).constraints
    if cons.stolen_resource is None:
        pytest.skip("no claimed resource: economic check auto-passes")
    floor = RESOURCE_COST_TABLE_USD[cons.stolen_resource]
    assert cons.min_amount_usd >= floor, (
        f"{path.stem}: min_amount_usd={cons.min_amount_usd} < floor={floor} "
        f"for {cons.stolen_resource.value}"
    )
    assert cons.max_amount_usd >= cons.min_amount_usd


def test_every_registered_spec_has_a_synthesizer():
    """SPEC_FILES and _SYNTHESIZERS must not drift apart."""
    assert set(SPEC_FILES) == set(_SYNTHESIZERS)


def test_sub_floor_band_is_rejected_at_load():
    """Regression: the exact ATTACK_6 misconfiguration must now fail loudly."""
    with pytest.raises(pydantic.ValidationError) as exc:
        OperationalConstraints(
            stolen_resource="synthetic_identity",
            pos_entry_modes=["CNP"],
            preferred_three_ds=["N"],
            target_verticals=["ecommerce"],
            min_amount_usd=180,
            max_amount_usd=1400,
        )
    assert "economic floor" in str(exc.value)


def test_inverted_band_is_rejected_at_load():
    with pytest.raises(pydantic.ValidationError):
        OperationalConstraints(
            stolen_resource="fullz",
            pos_entry_modes=["CNP"],
            preferred_three_ds=["N"],
            target_verticals=["ecommerce"],
            min_amount_usd=900,
            max_amount_usd=100,
        )
