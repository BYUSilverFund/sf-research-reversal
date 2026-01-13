import datetime as dt
from pathlib import Path

import altair as alt
import great_tables as gt
import polars as pl
import sf_quant.data as sfd

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "barra_reversal_volume"
gamma = 130
results_folder = Path("results/experiment_7")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Load MVO weights
weights = pl.read_parquet(f"weights/{signal_name}/{gamma}/*.parquet")

# Get returns
returns = (
    sfd.load_assets(
        start=start, end=end, columns=["date", "barrid", "return"], in_universe=True
    )
    .sort("date", "barrid")
    .select(
        "date",
        "barrid",
        pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return"),
    )
)

# Compute portfolio returns
portfolio_returns = (
    weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
    .sort("date")
)

# Compute cumulative log returns
cumulative_returns = portfolio_returns.select(
    "date", pl.col("return").log1p().cum_sum().mul(100).alias("cumulative_return")
)

# Plot cumulative log returns
chart = (
    alt.Chart(cumulative_returns, title="MVO Backtest Results (Active)")
    .mark_line()
    .encode(
        x=alt.X("date", title=""),
        y=alt.Y("cumulative_return", title="Cumulative Log Return (%)"),
    )
    .properties(width=800, height=400)
)

# Save chart
chart_path = results_folder / "cumulative_returns.png"
chart.save(chart_path, scale_factor=3)

# Create summary table
summary = portfolio_returns.select(
    pl.col("return").mean().mul(252).alias("mean_return"),
    pl.col("return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
).with_columns(pl.col("mean_return").truediv(pl.col("volatility")).alias("sharpe"))

table = (
    gt.GT(summary)
    .tab_header(title="MVO Backtest Results (Active)")
    .cols_label(
        mean_return="Mean Return",
        volatility="Volatility",
        sharpe="Sharpe",
    )
    .fmt_percent(["mean_return", "volatility"], decimals=2)
    .fmt_number("sharpe", decimals=2)
    .opt_stylize(style=4, color="gray")
)

table_path = results_folder / "summary_table.png"
table.save(table_path, scale=3)
