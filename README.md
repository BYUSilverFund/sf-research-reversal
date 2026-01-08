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
```

## Experiments
1. Standard reversal quantile backtest
2. Idiosyncratic + vol-scaled + smoothed reversal quantile backtest
3. Idiosyncratic + vol-scaled + smoothed reversal MVO backtest
4. Idiosyncratic + vol-scaled + smoothed reversal MVO portfolio check
5. Idiosyncratic + vol-scaled + smoothed reversal Windsorized MVO backtest
6. Idiosyncratic + vol-scaled + smoothed reversal Windsorized portfolio check
7. Idiosyncratic + vol-scaled + smoothed reversal Volume conditioned MVO backtest
8. Idiosyncratic + vol-scaled + smoothed reversal Volume conditioned portfolio check 

## Responsibilities
- Grant: 1
- Brandon: 2, 3
- Andrew: 4, 5, 6, 7, 8