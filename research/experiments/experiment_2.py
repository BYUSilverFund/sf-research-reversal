import datetime as dt

import polars as pl

from research.utils.data_processing import *
from research.utils.performance import *
from research.utils.quantile_backtest import *


def compute_barra_reversal(data: pl.DataFrame) -> pl.DataFrame:
    barra_reversal = data.with_columns(
        (
            pl.col("specific_return").ewm_mean(span=5, min_samples=5)
        )  # 5 day exponential average
        # .truediv((pl.col("specific_risk").rolling_mean(window_size=88, min_samples = 88))) # standardize returns
        .mul(-1)  # reversal methodology
        .shift(2)  # shift to make tradeable
        .over("barrid")
        .alias("barra_rev")
    ).sort(["barrid", "date"])

    return barra_reversal


if __name__ == "__main__":
    start = dt.date(1996, 1, 1)
    end = dt.date(2024, 12, 31)

    columns = [
        "date",
        "barrid",
        "ticker",
        "price",
        "return",
        "market_cap",
        "bid_ask_spread",
        "daily_volume",
        "specific_return",
        "specific_risk",
        "yield",
    ]

    russell = True
    price_filter = 5
    num_bins = 5
    results_path = (
        "/home/bwaits/Research/sf-research-reversal/research/experiments/results"
    )

    data = get_barra_data(start, end, columns, russell)
    barra_reversal = compute_barra_reversal(data)
    filtered_barra_reversal = filter_data(barra_reversal, "barra_rev", price_filter)
    bin_data_barra_reversal = compute_bins(
        filtered_barra_reversal, "barra_rev", num_bins
    )
    barra_reversal_portfolios = construct_equal_weight_portfolios(
        bin_data_barra_reversal, "barra_rev", num_bins
    )
    barra_reversal_portfolio_returns = compute_cumulative_log_returns(
        barra_reversal_portfolios, "barra_rev", num_bins
    )

    plot_quintile_portfolio_log_returns(
        barra_reversal_portfolios, "barra_rev", num_bins, results_path
    )
    calculate_quintile_summary_stats(
        barra_reversal_portfolio_returns, "barra_rev", results_path
    )
