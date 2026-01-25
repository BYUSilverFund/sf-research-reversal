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
