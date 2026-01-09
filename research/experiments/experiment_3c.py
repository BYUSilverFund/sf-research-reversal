
# sf_quant imports 
import sf_quant.optimizer as sfo


# import util functions 
from research.utils.data_processing import *
from research.utils.mvo_backtest import *
from research.utils.performance import *

# data processsing
import polars as pl

# built in
from pathlib import Path

if __name__ == "__main__":

    signals = ['barra_rev']
    
    constraint_type = 'zero_beta'
    for signal in signals:
        
        weights_folder = Path("/home/bwaits/Research/sf-research-reversal/research/experiments/weights")
        big_alpha_results_folder = Path("/home/bwaits/Research/sf-research-reversal/research/experiments/results")
        backtest_weights = pl.read_parquet(weights_folder / f"{signal}_*.parquet")
        print(backtest_weights)

        print(f"{signal} Zero Beta Backtest")
        construct_mvo_results(backtest_weights, signal, 'zero_beta', big_alpha_results_folder)

        







