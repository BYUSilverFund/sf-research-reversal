# Idiosyncratic + vol-scaled + smoothed reversal
# Volume conditioned MVO backtest

import datetime as dt

import polars as pl
import sf_quant.data as sfd
from dotenv import load_dotenv

from research.utils import run_backtest_parallel

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "barra_reversal_volume"
IC = 0.05
gamma = 130
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]

# Get data
data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "return",
        "specific_return",
        "specific_risk",
        "predicted_beta",
        "daily_volume",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

# Compute signal
signals = data.sort("barrid", "date").with_columns(
    pl.col("specific_return")
    .ewm_mean(span=5, min_samples=5)
    .mul(-1)
    .shift(1)
    .over("barrid")
    .alias(signal_name)
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

# Compute cross sectional z-scores for signal
scores = filtered.select(
    "date",
    "barrid",
    "price",
    "predicted_beta",
    "specific_risk",
    "daily_volume",
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)


# compute time-series z-score for the dollar volume of each security
volume_scores = (
    scores.sort("date", "barrid")
    .with_columns(
        dollar_volume=pl.col("daily_volume").mul(
            pl.col("price")
        )  # calculate dollar volume (I dont think this is necessary actually)
    )
    .with_columns(
        dollar_volume_mean=pl.col("dollar_volume")
        .rolling_mean(window_size=252, min_samples=252)
        .over("barrid"),
        dollar_volume_std=pl.col("dollar_volume")
        .rolling_std(window_size=252, min_samples=252)
        .over("barrid"),
    )
    .with_columns(
        pl.col("dollar_volume")
        .sub(pl.col("dollar_volume_mean"))
        .truediv(pl.col("dollar_volume_std"))
        .over("barrid")
        .alias("volume_score"),
    )
    # .with_columns(
    #     volume_score=pl.col("volume_score")
    #         .ewm_mean(span=3, ignore_nulls=True)
    #         .over("barrid")
    # )
    .sort(["barrid", "date"])
)


# Compute alphas with conditional logic: Set alpha to 0 if reversal is high with strong volume
alphas = (
    volume_scores.with_columns(
        # grinold and kahn alpha
        gk_alpha=pl.col("score") * IC * pl.col("specific_risk")
    )
    .with_columns(
        # Set alpha to 0 if both score > 1 and volume_score > 1
        alpha=pl.when((pl.col("score") > 1.0) & (pl.col("volume_score") > 1.0))
        .then(0.0)
        .otherwise(pl.col("gk_alpha"))
    )
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)


# Run parallelized backtest
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)
