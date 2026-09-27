from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "cmapss"


@pytest.fixture
def fixture_dir():
    """Excerpt of the real NASA C-MAPSS FD001 files: training and test engines
    1-3 and their official final RUL values."""
    return FIXTURES


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    from backend import main
    main._calls.clear()
    yield
    main._calls.clear()
