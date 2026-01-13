import datetime as dt

import polars as pl
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
from dotenv import load_dotenv
import great_tables as gt
from pathlib import Path


# Load environment variables
load_dotenv()

# Parameters
start = dt.date(2024, 1, 1)
end = dt.date(2025, 12, 30)
price_filter = 5
signal_name = "barra_reversal_volume"
results_folder = Path("results/experiment_8")
IC = 0.05
gamma = 10
n_cpus = 8
constraints = [
    sfo.FullInvestment(),
    sfo.LongOnly(),
    sfo.NoBuyingOnMargin(),
    sfo.UnitBeta()
]

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
    pl.col('date').eq(end)
)

# Compute scores
scores = (
    filtered
    .select(
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
)

# compute time-series z-score for the dollar volume of each security
volume_scores = (
    scores
    .sort('date', 'barrid')
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
        .sub(pl.col("dollar_volume").mean())
        .truediv(pl.col("dollar_volume").std())
        .over("barrid")
        .alias("volume_score"),
    )
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

# Set up mean variance optimization
alphas_np = alphas['alpha'].to_numpy()
betas_np = alphas['predicted_beta'].to_numpy()
barrids = alphas['barrid'].to_list()

# Get covariance matrix
covariance_matrix = sfd.construct_covariance_matrix(date_=end, barrids=barrids)
covariance_matrix_np = covariance_matrix.drop('barrid').to_numpy()

# Get optimal weights
weights = sfo.mve_optimizer(
    ids=barrids,
    alphas=alphas_np,
    covariance_matrix=covariance_matrix_np,
    constraints=constraints,
    gamma=gamma,
    betas=betas_np
)

# Get benchmark weights
benchmark_weights = (
    sfd.load_benchmark(
        start=start,
        end=end
    )
    .filter(
        pl.col('date').eq(end)
    )
)

# Get ticker mapping
tickers = sfd.load_assets_by_date(
    date_=end,
    in_universe=True,
    columns=['barrid', 'ticker']
)

# Compute active weights and pct_change_bmk
all_weights = (
    benchmark_weights
    .rename({'weight': 'bmk_weight'})
    .join(
        other=weights.rename({'weight': 'total_weight'}),
        on=['barrid'],
        how='left'
    )
    .join(
        other=tickers,
        on=['barrid'],
        how='left'
    )
    .with_columns(
        pl.col('total_weight').fill_null(0)
    )
    .select(
        'date',
        'ticker',
        'barrid',
        'total_weight',
        'bmk_weight',
        pl.col('total_weight').sub(pl.col('bmk_weight')).alias('active_weight')
    )
    .with_columns(
        pl.col('total_weight').truediv('bmk_weight').sub(1).alias('pct_change_bmk')
    )
    .sort('total_weight', descending=True)
    .head(10)
)

# Create summary table
table = (
    gt.GT(all_weights)
    .tab_header(title="Volume Conditioned Barra Reversal Portfolio")
    .cols_label(
        date="Date",
        ticker="Ticker",
        total_weight="Total Weight",
        bmk_weight="Benchmark Weight",
        active_weight="Active Weight",
        pct_change_bmk="Benchmark Percent Change"
    )
    .fmt_percent(["total_weight", "bmk_weight", "active_weight", "pct_change_bmk"], decimals=2)
    .opt_stylize(style=4, color="gray")
)

table_path = results_folder / "portfolio.png"
table.save(table_path, scale=3)