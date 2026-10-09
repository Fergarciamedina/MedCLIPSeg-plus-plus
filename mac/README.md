# Mac Environment (Apple Silicon)

This is a self-contained environment for running inference on a Mac.

## Setup
1. Create a virtual environment: `python3 -m venv .venv`
2. Activate it: `source .venv/bin/activate`
3. Install requirements: `pip install -r requirements.txt`

## Running
Execute `./run.sh`. It automatically sets data paths and calls `test.py` with `configs/BUSI_mac.yaml`.
Data is read from `project/data/raw/` and outputs go to `project/data/outputs/`.
