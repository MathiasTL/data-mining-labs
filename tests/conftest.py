import numpy as np
import polars as pl
import pytest


@pytest.fixture
def base_df() -> pl.DataFrame:
    """Synthetic table with the base-table contract: case_id, WEEK_NUM, target."""
    rng = np.random.default_rng(0)
    n = 400
    weeks = np.repeat(np.arange(0, 20), n // 20)
    signal = rng.normal(size=n)
    target = (signal + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    return pl.DataFrame(
        {
            "case_id": np.arange(n),
            "WEEK_NUM": weeks,
            "target": target,
            "x_signal": signal,
            "x_noise": rng.normal(size=n),
            "x_cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
