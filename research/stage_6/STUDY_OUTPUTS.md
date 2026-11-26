# Stage 6 · Outputs of the study and system

Location `research/stage_6/` · [Up to research/](../RESEARCH_GUIDE.md)

The study outputs are [aggregate tables and plots](../stage_4/experiments/RESULTS_GUIDE.md) and a frozen numeric policy. They answer how much of each test cohort is retained and how accurately the retained masks overlap their reference annotations. The [benchmarking notebook](../notebooks/02_scientific_benchmarking.ipynb) reads those artifacts and visualizes the results.

The dashboard presents the original scan, precomputed mask overlay and the decision for an individual request. If the query passes the policy and the request includes retrieval, the response can include similar eligible cases and their similarity scores. A scan without matching stored outputs receives **Not Evaluated**. The [application output guide](../../application/stages/stage6_system_output/DASHBOARD_OUTPUTS.md) describes these fields and the software checks behind them.

[Research overview](../RESEARCH_GUIDE.md) · [Study results](../stage_4/experiments/RESULTS_GUIDE.md)
