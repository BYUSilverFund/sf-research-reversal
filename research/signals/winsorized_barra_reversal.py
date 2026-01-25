import polars as pl


def barra_reversal() -> pl.Expr:
    return (
        pl.col("specific_return")
        .ewm_mean(span=5, min_samples=5)
        .mul(-1)
        .shift(1)
        .over("barrid")
        .alias("barra_reversal")
    )


def winsorized_barra_reversal() -> pl.Expr:
    return (
        pl.col("barra_reversal")
        .clip(
            lower_bound=pl.col("barra_reversal").quantile(0.025),
            upper_bound=pl.col("barra_reversal").quantile(0.975),
        )
        .over("date")
        .alias("winsorized_barra_reversal")
    )
