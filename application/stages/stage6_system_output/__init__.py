"""Stage 6: System Output Package."""
from .visual_composer import compose_system_output
from .triage_reporter import generate_triage_summary_report

__all__ = [
    "compose_system_output",
    "generate_triage_summary_report",
]
