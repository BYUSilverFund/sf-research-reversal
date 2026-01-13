import datetime as dt

import polars as pl
import sf_quant.data as sfd
from dotenv import load_dotenv

from research.utils import run_backtest_parallel

# Load environment variables
load_dotenv()

# Parameters
# start = dt.date(1996, 1, 1)
# end = dt.date(2024, 12, 31)
start = dt.date(2024, 12, 1)
end = dt.date(2024, 12, 31)

price_filter = 5
signal_name = "barra_reversal"
IC = 0.05
gamma = 400
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

# Compute alphas
alphas = (
    filtered.with_columns(
        pl.col(signal_name).mul(IC).mul("specific_risk").alias("alpha")
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
