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


def barra_reversal_score() -> pl.Expr:
    return (
        pl.col("barra_reversal")
        .sub(pl.col("barra_reversal").mean())
        .truediv(pl.col("barra_reversal").std())
        .over("date")
        .alias("barra_reversal_score")
    )


def winsorized_barra_reversal() -> pl.Expr:
    return (
        pl.col("barra_reversal_score")
        .clip(
            lower_bound=-2,
            upper_bound=2,
        )
        .over("date")
        .alias("winsorized_barra_reversal_score")
    )
