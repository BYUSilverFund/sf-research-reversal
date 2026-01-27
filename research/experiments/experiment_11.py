import datetime as dt
from pathlib import Path

import altair as alt
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

n_quantiles = 10
quantiles = [str(i) for i in range(n_quantiles)]
regression_data = alphas.join(
    other=forward_returns, on=["date", "barrid"], how="inner"
).with_columns(
    pl.col("alpha").qcut(n_quantiles, labels=quantiles).over("date").alias("quantile")
)


def fit_quantile_regression(regression_data: pl.DataFrame, quantiles: list[str]):
    results_list = []

    for quantile in quantiles:
        subset = regression_data.filter(pl.col("quantile").eq(quantile))

        model = smf.ols("fwd_return ~ alpha", subset).fit()
        conf_int = model.conf_int()

        results_list.append(
            pl.DataFrame(
                {
                    "quantile": quantile,
                    "coefficient": model.params["alpha"],
                    "ci_lower": conf_int.loc["alpha", 0],
                    "ci_upper": conf_int.loc["alpha", 1],
                }
            )
        )

    return pl.concat(results_list)


# Execute and visualize
results = fit_quantile_regression(regression_data, quantiles)

# Add error bars to show confidence intervals
chart = (
    alt.Chart(results)
    .mark_line(point=True)
    .encode(
        x=alt.X("quantile", title="Quantile"),
        y=alt.Y("coefficient", title="Alpha Coefficient"),
    )
    .properties(width=800, height=400)
)

# Add error bars for confidence interval
error_bars = (
    alt.Chart(results)
    .mark_errorbar()
    .encode(x="quantile", y=alt.Y("ci_lower", title="Alpha Coefficient"), y2="ci_upper")
)

# Save chart
chart_path = results_folder / "quantile_chart.png"
(error_bars + chart).save(chart_path, scale_factor=3)
