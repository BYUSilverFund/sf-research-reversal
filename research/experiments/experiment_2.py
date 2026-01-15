import datetime as dt
from pathlib import Path

import altair as alt
import great_tables as gt
import polars as pl
import seaborn as sns
import sf_quant.data as sfd
import statsmodels.formula.api as smf

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
num_bins = 5
signal_name = "barra_reversal"
results_folder = Path("results/experiment_2")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Get data
data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "specific_return",
    ],
    in_universe=True,
).with_columns(pl.col("specific_return").truediv(100))

# Compute signal
signals = data.sort("barrid", "date").with_columns(
    pl.col("specific_return")
    .ewm_mean(span=5, min_samples=5)
    .mul(-1)
    .shift(2)
    .over("barrid")
    .alias(signal_name)
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
)

# Create portfolios
labels = [str(i) for i in range(num_bins)]
portfolios = filtered.with_columns(
    pl.col(signal_name).qcut(num_bins, labels=labels).over("date").alias("bin")
)

# Compute portfolio returns
returns = (
    portfolios.group_by("date", "bin")
    .agg(pl.col("specific_return").mean())
    .with_columns(
        pl.col("specific_return")
        .truediv(
            pl.col("specific_return").std().mul(317)
        )  # Scale volatility to 5%: sqrt(252) / 5 = 3.17
        .over("bin")
    )
    .pivot(on="bin", index="date", values="specific_return")
    .with_columns(pl.col(str(num_bins - 1)).sub(pl.col("0")).alias("spread"))
    .unpivot(index="date", variable_name="bin", value_name="specific_return")
    .sort("date", "bin")
)

# Compute cumulative returns
cumulative_returns = returns.sort("date", "bin").select(
    "date",
    "bin",
    pl.col("specific_return")
    .log1p()
    .cum_sum()
    .mul(100)
    .over("bin")
    .alias("cumulative_return"),
)

# Plot cumulative log returns
colors = sns.color_palette("coolwarm", num_bins).as_hex()
colors.append("green")
chart = (
    alt.Chart(cumulative_returns, title="Quantile Backtest Results")
    .mark_line()
    .encode(
        x=alt.X("date", title=""),
        y=alt.Y("cumulative_return", title="Cumulative Log Return (%)"),
        color=alt.Color(
            "bin",
            title="Portfolio",
            scale=alt.Scale(domain=["0", "1", "2", "3", "4", "spread"], range=colors),
        ),
    )
    .properties(width=800, height=400)
)

# Save chart
chart_path = results_folder / "cumulative_returns.png"
chart.save(chart_path, scale_factor=3)

# Create summary table
summary = (
    returns.group_by("bin")
    .agg(
        pl.col("specific_return").mean().mul(252).alias("mean_return"),
        pl.col("specific_return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
    )
    .with_columns(pl.col("mean_return").truediv(pl.col("volatility")).alias("sharpe"))
    .sort("bin", descending=True)
)

table = (
    gt.GT(summary)
    .tab_header(title="Quantile Backtest Results")
    .cols_label(
        bin="Portfolio",
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

# Fama french regression
ff5 = (
    sfd.load_fama_french(start=start, end=end)
    .sort("date")
    .with_columns(pl.exclude("date").shift(-1))
)

regression_data = (
    returns.join(other=ff5, on="date", how="left")
    .drop_nulls("specific_return")
    .with_columns(pl.col("specific_return").sub("rf").alias("specific_return_rf"))
    .with_columns(pl.exclude("date", "bin").mul(100))
)

bins = labels + ["spread"]
regression_summary_list = []
for bin in bins:
    regression_data_slice = regression_data.filter(pl.col("bin").eq(bin))
    formula = "specific_return_rf ~ mkt_rf + smb + hml + rmw + cma"
    model = smf.ols(formula, regression_data_slice)
    results = model.fit()

    regression_summary = pl.DataFrame(
        {
            "variable": results.params.index,
            "coefficient": results.params.values,
            "tstat": results.tvalues.values,
        }
    ).with_columns(pl.lit(bin).alias("bin"))

    regression_summary_list.append(regression_summary)

regression_summary = (
    pl.concat(regression_summary_list)
    .sort("bin")
    .pivot(index="bin", on="variable", values=["coefficient", "tstat"])
)

regression_table = (
    gt.GT(regression_summary)
    .tab_header(title="Quantile Backtest Results (Daily %)")
    .cols_label(
        bin="Portfolio",
        coefficient_Intercept="Intercept Coef.",
        coefficient_mkt_rf="MKT Coef.",
        coefficient_smb="SMB Coef.",
        coefficient_hml="HML Coef.",
        coefficient_rmw="RMW Coef.",
        coefficient_cma="CMA Coef.",
        tstat_Intercept="Intercept T-stat",
        tstat_mkt_rf="MKT T-stat",
        tstat_smb="SMB T-stat",
        tstat_hml="HML T-stat",
        tstat_rmw="RMW T-stat",
        tstat_cma="CMA T-stat",
    )
    .fmt_number(pl.exclude("bin"), decimals=4)
    .opt_stylize(style=4, color="gray")
)

table_path = results_folder / "regression_table.png"
regression_table.save(table_path, scale=3)
