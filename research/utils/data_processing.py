import sf_quant.data as sfd
import polars as pl
import datetime as dt


def get_barra_data(start: dt.date, end: dt.date, columns: list[str], russell: bool):

    # data loader
    data = sfd.load_assets(
        start=start,
        end=end,
        in_universe=russell, # bool to indicate if we are only using Russell 3000 stocks
        columns=columns
    )

    # convert returns to decimal space
    data = (
        data.with_columns(
            pl.col('return').truediv(100).alias('return')))

    data = (
        data.with_columns(
            pl.col('specific_return').truediv(100).alias('specific_return')))

    data = (
        data.with_columns(
            pl.col('specific_risk').truediv(100).alias('specific_risk')
        )
    )

    data = (
        data.with_columns(
            pl.col("daily_volume").replace(0, None)
        )
    )

    if not russell:
        # filter out securities we dont want to consider if we are using stocks outside of Russell 3000
        data = data.filter(pl.col('iso_country_code').eq("USA"), pl.col('rootid').eq(pl.col('barrid')), pl.col('barrid').str.starts_with('US')) #pl.col('price').ge(5))

    return data


def filter_prices(data: pl.DataFrame, signal: pl.DataFrame, lag: bool, price_filter: pl.Int64 | None = None, market_cap_filter: pl.Int64 | None = None) -> pl.DataFrame:

    # filter out securities which will cause errors in portfolio calculuations such as null/NaN values
    # filter out low-price securities (penny stocks) as they are often non tradeable or skew results. 

    data = data.sort(["barrid", "date"]).with_columns(
            pl.col("price")
            .shift(1)
            .over("barrid")
            .alias("price_lag")
            )
    
    data = data.sort(["barrid", "date"]).with_columns(
            pl.col("market_cap")
            .shift(1)
            .over("barrid")
            .alias("market_cap_lag")
            )
    

    # filter out no signal data
    data = data.filter(pl.col(signal).is_not_null(), pl.col(signal).is_not_nan())

    # filter out no alpha data if alpha column exists
    if 'alpha' in data.columns:
        data = data.filter(
            pl.col('alpha').is_not_null(), 
            pl.col('alpha').is_not_nan()
        )

    # filter based on parameters
    if lag:
        if price_filter:
            data = data.filter(pl.col("price_lag") >= price_filter)
        if market_cap_filter:
            data = data.filter(pl.col("market_cap_lag") >= market_cap_filter)

    else:
        if price_filter:
            data = data.filter(pl.col("price") >= price_filter)
        if market_cap_filter:
            data = data.filter(pl.col("market_cap") >= market_cap_filter)
    
    return data