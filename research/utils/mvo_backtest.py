import polars as pl
import sf_quant.backtester as sfb


def compute_alphas(data: pl.DataFrame, signal: str) -> pl.DataFrame:
    # compute Grinold & Kahn cross sectional alphas
    alphas = (
        data.with_columns(
            (
                (
                    pl.col(f"{signal}").sub(pl.col(f"{signal}").mean().over("date"))
                ).truediv(pl.col(f"{signal}").std().over("date"))
            ).alias("score")
        )
        .with_columns(
            (pl.col("score").mul(pl.col("specific_risk")).mul(0.05)).alias(
                f"{signal}_alpha"
            )
        )
        .sort(["barrid", "date"])
    )

    return alphas


def compute_mvo_backtest(data: pl.DataFrame, constraints, gamma) -> pl.DataFrame:
    # run parallel MVO backtest with sf_quant
    weights = sfb.backtest_parallel(
        data=data, constraints=constraints, gamma=gamma, n_cpus=32
    )

    return weights
