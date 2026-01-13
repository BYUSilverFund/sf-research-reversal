import polars as pl


def compute_bins(data: pl.DataFrame, signal: str, num_bins: pl.Int64) -> pl.DataFrame:
    # bin the signal for quantile/decile and spread portfolio construction
    labels = [str(i) for i in range(num_bins)]

    bins = data.with_columns(
        pl.col(signal)
        .qcut(num_bins, labels=labels, allow_duplicates=True)
        .over("date")
        .alias(f"{signal}_bin")
    ).sort(["barrid", "date"])
    return bins


def construct_equal_weight_portfolios(
    bins: pl.DataFrame, signal1: str, num_bins: pl.Int64
) -> pl.DataFrame:
    # construct equal-weight portfolio within each bin
    ew_port = (
        bins.group_by(["date", f"{signal1}_bin"])
        .agg(pl.col("return").mean())
        .pivot(on=f"{signal1}_bin", index="date", values="return")
        # force spread portfolio to use equal ex-post vol of high and low bins
        .with_columns(
            (
                pl.col(f"{num_bins - 1}")
                - pl.col("0")
                * (pl.col(f"{num_bins - 1}").std().truediv(pl.col("0").std()))
            ).alias("spread")
        )
        # no change to spread portfolio
        # .with_columns((pl.col(f'{num_bins-1}') - pl.col('0')).alias('spread'))
        .sort("date")
    )

    return ew_port


def compute_cumulative_log_returns(
    portfolio: pl.DataFrame, signal: str, num_bins: pl.Int64
) -> pl.DataFrame:
    # compute cumulative return metrics for plotting and calculating statistics
    portfolio_returns = (
        portfolio.unpivot(
            index="date",
            on=[str(i) for i in range(num_bins)] + ["spread"],
            variable_name=f"{signal}_bin",
            value_name="return",
        )
        .with_columns(
            [
                (pl.col("return") * 100).alias("return_pct"),
                pl.col("return").log1p().cum_sum().alias("cum_log_return"),
            ]
        )
        .sort("date")
    )

    return portfolio_returns
