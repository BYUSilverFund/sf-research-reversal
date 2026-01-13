import datetime as dt
import os
import subprocess
import tempfile

import polars as pl
import sf_quant.data as sfd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parameters
# start = dt.date(1996, 1, 1)
start = dt.date(2022, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "barra_reversal"
IC = 0.05
gamma = 400

# Get data
data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "return",
        "specific_return",
        "specific_risk",
        "predicted_beta",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

# Compute signal
signals = data.sort("barrid", "date").with_columns(
    pl.col("specific_return")
    .ewm_mean(span=5, min_samples=5)
    .mul(-1)
    .shift(1)
    .over("barrid")
    .alias(signal_name)
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

# Compute alphas
alphas = (
    filtered.with_columns(
        pl.col(signal_name).mul(IC).mul("specific_risk").alias("alpha")
    )
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

# Get unique years from the alphas data
years = sorted(alphas.select(pl.col("date").dt.year()).unique().to_series().to_list())
num_years = len(years)

# Get super computer job variables
byu_email = os.getenv("BYU_EMAIL")
project_root = os.getenv("PROJECT_ROOT")
years_str = " ".join(str(y) for y in years)
temp_dir = f"{project_root}/temp"
data_path = f"{temp_dir}/alphas.parquet"
output_dir = f"{project_root}/weights/{signal_name}/{gamma}/"

# Save alphas to temporary directory
os.makedirs(temp_dir, exist_ok=True)
alphas.write_parquet(data_path)

# Create logs directory
logs_dir = f"logs/{signal_name}/{gamma}"
os.makedirs(logs_dir, exist_ok=True)

# Format sbatch_script
sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=reversal_backtest
#SBATCH --output=logs/{signal_name}/{gamma}/backtest_%A_%a.out
#SBATCH --error=logs/{signal_name}/{gamma}/backtest_%A_%a.err
#SBATCH --array=0-{num_years - 1}%31
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-user={byu_email}
#SBATCH --mail-type=BEGIN,END,FAIL

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

DATA_PATH="{data_path}"
OUTPUT_DIR="{output_dir}"
GAMMA="{gamma}"

# Years to process
years=({years_str})

num_years=${{#years[@]}}

if [ $SLURM_ARRAY_TASK_ID -ge $num_years ]; then
  echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (max $((num_years-1)))."
  exit 1
fi

year=${{years[$SLURM_ARRAY_TASK_ID]}}

source {project_root}/.venv/bin/activate
echo "Running year=$year"
srun python research/utils/mvo.py --data_path "$DATA_PATH" --gamma "$GAMMA" --year "$year" --output_dir "$OUTPUT_DIR"
"""

# Write the script to a temporary file and submit it
with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
    f.write(sbatch_script)
    script_path = f.name

try:
    # Submit the job using sbatch
    result = subprocess.run(
        ["sbatch", script_path], capture_output=True, text=True, check=True
    )
    print(f"Job submitted successfully!")
    print(f"sbatch output: {result.stdout}")
    if result.stderr:
        print(f"sbatch stderr: {result.stderr}")
except subprocess.CalledProcessError as e:
    print(f"Error submitting job: {e}")
    print(f"stdout: {e.stdout}")
    print(f"stderr: {e.stderr}")
except FileNotFoundError:
    print(
        "Error: sbatch command not found. Are you running this on a system with SLURM?"
    )
finally:
    # Clean up the temporary file
    if os.path.exists(script_path):
        os.unlink(script_path)
