import sf_quant.performance as sfp

import polars as pl
import pandas as pd

import numpy as np
import statsmodels.formula.api as smf

import seaborn as sns
import matplotlib.pyplot as plt

import os
from pathlib import Path


def plot_quintile_portfolio_log_returns(portfolio_returns: pl.DataFrame, signal: str, num_bins: pl.Int64, path = str):

    # bin numbering
    bins_labels = [str(i) for i in range(num_bins)] + ["spread"]

    # compute cumulative log returns for each portfolio column
    cumulative_portfolio_returns = (
        portfolio_returns
        .with_columns([
            pl.col(c).log1p().cum_sum().alias(c) for c in bins_labels
        ])
        .select(["date"] + bins_labels)
        .sort("date")
    )

    # construct backtest plot
    plt.figure(figsize=(10, 6))
    plot_labels = [str(i) for i in range(num_bins)]
    colors = sns.color_palette('coolwarm', n_colors=len(plot_labels))

    # plot backtests
    for color, label in zip(colors, plot_labels): # quintile portfolios
        sns.lineplot(cumulative_portfolio_returns, x='date', y=label, label=label, color = color)

    # spread portfolio
    sns.lineplot(cumulative_portfolio_returns, x='date', y='spread', color = 'black', label='Spread')

    # plot labeling and saving figure 
    plt.ylabel("Cumulative Log Returns")
    plt.title(f'Decile Portfolios for {signal}')
    plt.legend()
    # plt.show()

    folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder / f"{signal}_plot.png")
    
    return


def calculate_quintile_summary_stats(port_returns: pl.DataFrame, signal: str, path: str):
    # calculate annualized portfolio metrics for each quintile portfolio
    stats = (
        port_returns
        .group_by(f"{signal}_bin")
        .agg([
            (pl.col("return").mean() * 252).alias("avg_return_ann"), # annualized returns
            (pl.col("return").std() * np.sqrt(252)).alias("vol_ann") # annualized volatility
        ])
        .with_columns(
            (pl.col("avg_return_ann") / pl.col("vol_ann")).alias("sharpe_ann") # annualized sharpe
        )
        .sort(f"{signal}_bin")
    )

    # Save metrics
    stats_df = stats.to_pandas()
    folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
    os.makedirs(folder, exist_ok=True)
    stats_df.to_parquet(folder / f"{signal}_decile_backtest.parquet")

    print(f'{signal} Stats')
    print(stats_df)
    print()

    return 


def construct_mvo_results(weights: pl.DataFrame, signal: str, constraint_type: str, path: str):
    
    folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
    os.makedirs(folder, exist_ok=True)

    # get MVO portfolio returns
    returns = sfp.generate_returns_from_weights(weights=weights)

    # save backtest results
    returns.write_parquet(folder / f"{signal}_{constraint_type}_mvo_backtest_data.parquet")

    # generate backtest plot
    portfolio_returns = sfp.generate_returns_chart(returns = returns,
                            title=f"{signal} {constraint_type} MVO Backtest",
                            log_scale=True,
                            file_name=folder / f"{signal}_{constraint_type}_mvo_backtest.png")
    

    # generate MVO portolio metrics
    summary = sfp.generate_summary_table(
        returns = returns
    )
    
    # save and display metrics
    summary.write_parquet(folder / f"{signal}_{constraint_type}_mvo_backtest_summary.parquet")

    print(f'{signal} MVO summary')
    print(summary)
    print()


# Here are functions for Fama-French Regressions. (Unsure how we want to import the correct data)

# def calculate_ff_regression(data: pl.DataFrame, signal: str):
#     factors = pl.read_csv("/home/bwaits/Research/Waits-Research/labs/ff6_daily.csv").cast({"date": pl.Date})
#     port = data.clone()

#     # add the factor portfolios to portfolio dataframe
#     port = port.join(factors, on="date", how="left").drop_nulls()

#     port = port.to_pandas() # convert to pandas to be compatible with statsmodels
#     port = port.set_index("date")

#     # run multivariate regression, testing the ff3 + data
#     # x_variables = '~ 1 + mktrf + smb + hml + umd + cma + rmw'
#     x_variables = '~ 1 + mktrf + smb + hml + umd'
#     # reg = smf.ols("spread" + x_variables, data=port).fit().summary()
#     # reg = reg.tables[1]

#     model = smf.ols("spread" + x_variables, data=port).fit()
 
#     ci = model.conf_int()
#     reg_pd = pd.DataFrame({
#         "term": model.params.index,
#         "coef": model.params.values,
#         "std_err": model.bse.values,
#         "t": model.tvalues.values,
#         "p": model.pvalues.values,
#         "ci_lo": ci[0].values,
#         "ci_hi": ci[1].values,
#     })


#     reg_pl = pl.from_pandas(reg_pd.reset_index(drop=True))
    

#     folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
#     os.makedirs(folder, exist_ok=True)
#     reg_pl.write_parquet(folder / f"{signal}_factor_regression.parquet")

#     print(f'{signal} Regression Stats')
#     print(reg_pl)
#     print()


# def calculate_zero_beta_ff_regression(data: pl.DataFrame, signal: str):
#     factors = pl.read_csv("/home/bwaits/Research/Waits-Research/labs/ff6_daily.csv").cast({"date": pl.Date})
#     port = data.clone()

#     # add the factor portfolios to portfolio dataframe
#     port = port.join(factors, on="date", how="left").drop_nulls()

#     port = port.to_pandas() # convert to pandas to be compatible with statsmodels
#     port = port.set_index("date")
#     print(port)

#     # run multivariate regression, testing the ff3 + data
#     # x_variables = '~ 1 + mktrf + smb + hml + umd + cma + rmw'
#     x_variables = '~ 1 + mktrf + smb + hml + umd'
#     # reg = smf.ols("spread" + x_variables, data=port).fit().summary()
#     # reg = reg.tables[1]

#     model = smf.ols("pnl" + x_variables, data=port).fit()
 
#     ci = model.conf_int()
#     reg_pd = pd.DataFrame({
#         "term": model.params.index,
#         "coef": model.params.values,
#         "std_err": model.bse.values,
#         "t": model.tvalues.values,
#         "p": model.pvalues.values,
#         "ci_lo": ci[0].values,
#         "ci_hi": ci[1].values,
#     })


#     reg_pl = pl.from_pandas(reg_pd.reset_index(drop=True))
    
#     folder = Path("/home/bwaits/Research/Waits-Research/labs/illiquidity_results")
#     os.makedirs(folder, exist_ok=True)
#     reg_pl.write_parquet(folder / f"{signal}_zero_beta_factor_regression.parquet")

#     print(f'{signal} Regression Stats')
#     print(reg_pl)
#     print()

#     # coefficients
#     alpha_hat = float(model.params['Intercept'])
#     beta_hat = model.params.drop('Intercept')

#     # build risk component
#     factor_cols = list(beta_hat.index)
#     risk_component = (port[factor_cols] * beta_hat.values).sum(axis=1)

#     # vector of alpha + residual (same as pnl - risk_component)
#     alpha_plus_resid = alpha_hat + model.resid
#     risk_adjusted_pnl = port['pnl'] - risk_component  

#     # adjusted pnl metrics
#     adj_df = pd.DataFrame({
#         'date': port.index,
#         'alpha_plus_resid': alpha_plus_resid,
#         'risk_adjusted_pnl': risk_adjusted_pnl,  
#     }).set_index('date')

#     print(adj_df)

#     # back to Polars and save
#     adj_pl = pl.from_pandas(adj_df.reset_index())
#     adj_pl.write_parquet(folder / f"{signal}_alpha_plus_residual_timeseries.parquet")
    
#     return adj_pl
