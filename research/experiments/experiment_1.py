# Quantile and MVO Reversal Backtest

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import sf_quant.data as sfd
from matplotlib import pyplot as plt

# Define start and end date
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
num_bins = 5
signal_name = "reversal"
results_folder = Path("results/experiment_1")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Load data
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
).with_columns(pl.col("return", "specific_return", "specific_risk").truediv(100))

# Create signal
signals = data.with_columns(
    pl.col("return")
    .log1p()
    .rolling_sum(window_size=21)
    .mul(-1)
    .shift(1)
    .over("barrid")
    .alias(signal_name)
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
)

# Create quintile portfolios
labels = [str(i) for i in range(num_bins)]
portfolios = filtered.with_columns(
    pl.col(signal_name)
    .rank(method="ordinal")
    .qcut(num_bins, labels=labels, allow_duplicates=True)
    .over("date")
    .alias("bin")
)

# Create reversal portfolios
returns = (
    portfolios.group_by("date", "bin")
    .agg(pl.col("return").mean())
    .pivot(on="bin", index="date", values="return")
    .with_columns(pl.col(str(num_bins - 1)).sub(pl.col("0")).alias("spread"))
    .unpivot(index="date", variable_name="bin", value_name="return")
    .sort("date", "bin")
)

# Compute cumulative returns
cumulative_returns = returns.sort("date", "bin").select(
    "date",
    "bin",
    pl.col("return").log1p().cum_sum().mul(100).over("bin").alias("cumulative_return"),
)

# Plot cumulative log returns
plot_df = cumulative_returns.sort("date", "bin").pivot(
    on="bin", index="date", values="cumulative_return"
)

plt.figure(figsize=(10, 6))
colors = sns.color_palette("coolwarm", num_bins).as_hex()
colors.append("green")
x_data = plot_df["date"].to_numpy()
bin_list = [str(j) for j in range(num_bins)] + ["spread"]

for i, label in enumerate(bin_list):
    y_data = plot_df[label].to_numpy()
    sns.lineplot(x=x_data, y=y_data, label=label, color=colors[i])

plt.title("Short-Term Reversal Quantile Backtest")
plt.ylabel("Cumulative Log Return (%)")
plt.legend(title="Portfolio", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()

# Save chart
chart_path = results_folder / "cumulative_returns.png"
plt.savefig(chart_path, dpi=300)

summary = (
    returns.group_by("bin")
    .agg(
        mean_return=pl.col("return").mean().mul(252),
        volatility=pl.col("return").std().mul(np.sqrt(252)),
    )
    .with_columns((pl.col("mean_return").truediv(pl.col("volatility"))).alias("sharpe"))
    .sort("bin", descending=True)
)

plt.figure(figsize=(12, 6))
plt.axis("off")

table_data = [summary.columns] + summary.with_columns(
    pl.col(pl.Float64).round(4)
).to_numpy().tolist()

the_table = plt.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
    colWidths=[0.15] * len(summary.columns),
)
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.2)

table_path = results_folder / "summary_table.png"
plt.savefig(table_path, dpi=300, bbox_inches="tight")
