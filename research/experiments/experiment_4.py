import datetime as dt

import polars as pl
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Parameters
end = dt.date(2025, 12, 30)
window = 21
start = end - dt.timedelta(days=window)
price_filter = 5
signal_name = "barra_reversal"
IC = 0.05
gamma = 10
n_cpus = 8
constraints = [
    sfo.FullInvestment(),
    sfo.LongOnly(),
    sfo.NoBuyingOnMargin(),
    sfo.UnitBeta()
]

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
    pl.col('date').eq(end)
)

# Compute scores
scores = (
    filtered
    .select(
        'date',
        'barrid',
        'predicted_beta',
        'specific_risk',
        pl.col(signal_name).sub(pl.col(signal_name).mean()).truediv(pl.col(signal_name).std()).over('date').alias('score'),
    )
)

# Compute alphas
alphas = (
    scores.with_columns(
        pl.col('score').mul(IC).mul("specific_risk").alias("alpha")
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
        pl.col('total_weight').truediv('bmk_weight').alias('pct_change_bmk')
    )
    .sort('total_weight', descending=True)
)

print(all_weights.head(10))
