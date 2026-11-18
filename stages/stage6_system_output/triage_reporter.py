"""Stage 5: Triage Summary Reporter (MOD-CLINICAL-PRESENT / L1.6).

Formats clinical batch processing summaries, telemetry statistics, and audit records.
"""

from typing import List, Dict, Any
import pandas as pd

def generate_triage_summary_report(cases_telemetry: List[Dict[str, Any]]) -> pd.DataFrame:
    """Produces a structured clinical review table."""
    records = []
    for c in cases_telemetry:
        records.append({
            "Case ID": c.get("image_id", "Unknown"),
            "Clinical Status": c.get("status", "Unknown"),
            "U_fg (nats)": round(c.get("u_fg", float("nan")), 5),
            "tau* Threshold": 0.064058,
            "Safety Margin": round(c.get("margin", 0.0), 5),
            "Foreground Pixels": c.get("fg_pixels", 0),
            "Retrieved Cases": len(c.get("retrieved_neighbors", [])),
            "Routing Action": "Autonomous Retrieval" if "ACCEPTED" in c.get("status", "") else "Radiologist Audit Queue",
        })
    return pd.DataFrame(records)

