import datetime as dt

import polars as pl
import sf_quant.data as sfd


def get_barra_data(start: dt.date, end: dt.date, columns: list[str], russell: bool):
    # data loader
    data = sfd.load_assets(
        start=start,
        end=end,
        in_universe=russell,  # bool to indicate if we are only using Russell 3000 stocks
        columns=columns,
    )

    # convert returns to decimal space
    data = data.with_columns(
        pl.col("return").truediv(100).alias("return"),
        pl.col("specific_return").truediv(100).alias("specific_return"),
        pl.col("specific_risk").truediv(100).alias("specific_risk"),
    )

    # filter out invalid volume entries
    data = data.filter(
        pl.col("daily_volume").is_not_null() & (pl.col("daily_volume") > 1)
    )

    return data


def filter_data(
    data: pl.DataFrame, signal: pl.DataFrame, price_filter: pl.Int64 | None = None
) -> pl.DataFrame:
    # filter out securities which will cause errors in portfolio calculuations such as null/NaN values
    # filter out low-price securities (penny stocks) as they are often non tradeable or skew results.

    # calculate lagged price
    data = data.sort(["barrid", "date"]).with_columns(
        pl.col("price").shift(1).over("barrid").alias("price_lag")
    )

    # filter out no signal data
    data = data.filter(pl.col(signal).is_not_null(), pl.col(signal).is_not_nan())

    # filter based on parameters
    if price_filter:
        data = data.filter(pl.col("price_lag") >= price_filter)

    return data
