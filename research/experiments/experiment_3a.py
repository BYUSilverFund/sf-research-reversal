# Idiosyncratic + vol-scaled + smoothed reversal MVO backtest

# import util functions 
from research.utils.data_processing import *
from research.utils.mvo_backtest import *
from research.utils.performance import *

# data processing
import polars as pl

# built in packages
import datetime as dt


def compute_barra_reversal(data: pl.DataFrame) -> pl.DataFrame:

    barra_reversal = (
        data.with_columns(
            (pl.col("specific_return").ewm_mean(span=5, min_samples = 5)) # 5 day exponential average
            # .truediv((pl.col("specific_risk").rolling_mean(window_size=88, min_samples = 88))) # standardize returns
            .mul(-1) # reversal methodology
            .shift(2) # shift to make tradeable
            .over('barrid')
            .alias('barra_rev')
        )
        .sort(['barrid', 'date'])
    )

    return barra_reversal

if __name__ == "__main__":

    start = dt.date(1995, 6, 27)
    end = dt.date(2025, 11, 12)

    # data fields and filtering parameters
    columns = [
        'date',
        'barrid',
        'ticker',
        'price',
        'return',
        'predicted_beta',
        'market_cap', 
        'bid_ask_spread',
        'daily_volume',
        'specific_return',
        'specific_risk',
        'yield'
    ]

    russell = True
    price_filter = 5

    # where to save alphas
    alpha_path = "/home/bwaits/Research/sf-research-reversal/research/experiments/alphas"

    data = get_barra_data(start, end, columns, russell)
    barra_reversal = compute_barra_reversal(data)
    filtered_barra_reversal = filter_data(barra_reversal, 'barra_rev', price_filter)

    signals = ['barra_rev']
    signals_str = '_'.join(signals)

    for signal in signals:
        filtered_barra_reversal = compute_alphas(filtered_barra_reversal, signal)

   
    folder = Path(alpha_path)
    os.makedirs(folder, exist_ok=True)
    filtered_barra_reversal.write_parquet(folder / f"{signals_str}.parquet")



