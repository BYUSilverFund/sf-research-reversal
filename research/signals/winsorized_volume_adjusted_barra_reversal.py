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


def dollar_volume() -> pl.Expr:
    return pl.col("daily_volume").mul(pl.col("price")).log1p().alias("dollar_volume")


def dollar_volume_score() -> pl.Expr:
    return (
        pl.col("dollar_volume")
        .rolling_mean(window_size=252, min_samples=1)
        .truediv(pl.col("dollar_volume").rolling_std(window_size=252, min_samples=2))
        .over("barrid")
        .alias("dolar_volume_score")
    )


def volume_adjusted_barra_reversal() -> pl.Expr:
    return (
        pl.when(
            pl.col("barra_reversal_score").gt(2) & pl.col("dollar_volume_score").gt(2)
        )
        .then(pl.lit(0))
        .otherwise(pl.col("barra_reversal"))
        .alias("volume_adjusted_barra_reversal")
    )


def winsorized_volume_adjusted_barra_reversal() -> pl.Expr:
    return (
        pl.col("volume_adjusted_barra_reversal")
        .clip(
            lower_bound=pl.col("volume_adjusted_barra_reversal").quantile(0.025),
            upper_bound=pl.col("volume_adjusted_barra_reversal").quantile(0.975),
        )
        .over("date")
        .alias("winsorized_volume_adjusted_barra_reversal")
    )
