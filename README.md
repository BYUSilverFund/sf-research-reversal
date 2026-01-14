# Silver Fund Reversal Research Repository

## Set Up

Set up your Python virtual environment using `uv`.
```bash
uv sync
```

Source your Python virtual environment.
```bash
source .venv/bin/activate
```

Set up your environment variables in a `.env` file. You can follow the example found in `.env.example`.
```
ASSETS_TABLE=
EXPOSURES_TABLE=
COVARIANCES_TABLE=
CRSP_DAILY_TABLE=
CRSP_MONTHLY_TABLE=
CRSP_EVENTS_TABLE=
BYU_EMAIL=
PROJECT_ROOT=
```

Set up pre-commit by running:
```bash
prek install
```

Now all of your files will be formatted on commit (you will need to re-commit after the formatting).

## Experiments
1. Standard reversal quantile backtest
2. Idiosyncratic + vol-scaled + smoothed reversal quantile backtest
3. Idiosyncratic + vol-scaled + smoothed reversal MVO backtest
4. Idiosyncratic + vol-scaled + smoothed reversal MVO portfolio check
5. Idiosyncratic + vol-scaled + smoothed reversal Windsorized MVO backtest
6. Idiosyncratic + vol-scaled + smoothed reversal Windsorized portfolio check
7. Idiosyncratic + vol-scaled + smoothed reversal Volume conditioned MVO backtest
8. Idiosyncratic + vol-scaled + smoothed reversal Volume conditioned portfolio check 
9. Standard reversal MVO backtest
10. Standard reversal portfolio check
11. Quantile regression of idiosyncratic + vol-scaled + smoothed reversal signal vs. returns
12. 

## To do
- Experiments 9 and 10
- Add FF5 regressions to 1, 2, 5, 7, 9
- Add IC chart to 5, 7, 9
- Add vol- and beta- adjusted backtests to 1 and 2

## Responsibilities
- Grant: 1
- Brandon: 2, 5, 7, 9, 10
- Andrew: 3, 4, 6, 8, 11
