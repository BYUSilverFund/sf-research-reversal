import polars as pl


def reversal() -> pl.Expr:
    return (
        pl.col("return")
        .log1p()
        .rolling_sum(21)
        .mul(-1)
        .over("barrid")
        .alias("reversal")
    )
