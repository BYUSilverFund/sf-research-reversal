import datetime as dt
from pathlib import Path

import great_tables as gt
import polars as pl
import sf_quant.data as sfd
import statsmodels.formula.api as smf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "barra_reversal"
IC = 0.05
results_folder = Path("results/experiment_11")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Get data
data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "price",
        "return",
        "specific_return",
        "specific_risk",
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
    pl.col("specific_risk").is_not_null(),
)

# Compute scores
scores = filtered.select(
    "date",
    "barrid",
    "specific_risk",
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)

# Compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha")
    .sort("date", "barrid")
)

# Get forward returns
forward_returns = (
    data.sort("date", "barrid")
    .select(
        "date", "barrid", pl.col("return").shift(-1).over("barrid").alias("fwd_return")
    )
    .drop_nulls("fwd_return")
)

# Merge alphas and forward returns
regression_data = (
    alphas.join(other=forward_returns, on=["date", "barrid"], how="inner")
    .select("date", "barrid", "alpha", "fwd_return")
    .with_columns(pl.col("alpha", "fwd_return").mul(100))
    .filter(pl.col("date").eq(pl.col("date").max()))
)

# Add percentile rank for alpha
regression_data = regression_data.with_columns(
    pl.col("alpha").rank(method="average").truediv(pl.len()).mul(100).alias("alpha_pct")
)

# Run regressions for different distribution segments
segment_results = []
for lower_pct in range(90, 100, 1):
    upper_pct = lower_pct + 1

    # Filter data for this segment
    segment_data = regression_data.filter(
        (pl.col("alpha_pct") >= lower_pct) & (pl.col("alpha_pct") < upper_pct)
    )

    # Run regression
    if len(segment_data) > 0:
        model = smf.ols("fwd_return ~ alpha", data=segment_data)
        results = model.fit()

        # Extract coefficient and t-stat for alpha
        alpha_coef = results.params["alpha"]
        alpha_tstat = results.tvalues["alpha"]
        n_obs = len(segment_data)

        segment_results.append(
            {
                "segment": f"{lower_pct}-{upper_pct}",
                "alpha_coef": alpha_coef,
                "alpha_tstat": alpha_tstat,
                "n_obs": n_obs,
            }
        )

# Create polars dataframe with results
results_df = pl.DataFrame(segment_results)

table = (
    gt.GT(results_df)
    .tab_header(title="Quantile Regression Results")
    .cols_label(
        segment="Segment",
        alpha_coef="Alpha Coefficient",
        alpha_tstat="Alpha T-stat",
        n_obs="# Observations",
    )
    .fmt_number(["alpha_coef", "alpha_tstat"], decimals=3)
    .fmt_integer("n_obs")
    .opt_stylize(style=4, color="gray")
)

table_path = results_folder / "quantile_regression_table.png"
table.save(table_path, scale=3)
