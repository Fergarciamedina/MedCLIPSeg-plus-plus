"""Stage 2: Intent Router (MOD-INTENT-ROUTER / Stage 2).

Parses clinical text request R via f_intent(R) to produce an Execution Plan:
- s in {0, 1}: Segment flag
- r in {0, 1}: Retrieve flag
- K in N: Number of top-K cases to retrieve (default K=5)

Supports three operational routing paths:
- s=1, r=0: Segment only
- s=1, r=1: Both (Segment + Retrieve)
- s=0, r=1: Retrieve only
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re

@dataclass
class ExecutionPlan:
    """Stage 2 Execution Plan structure."""
    raw_prompt: str
    s: int  # Segment flag: 1 or 0
    r: int  # Retrieve flag: 1 or 0
    k: int = 5  # Top-K cases to retrieve
    target_concept: str = "breast lesion"
    filter_status: str = "Accepted"
    dataset_filters: List[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if self.s == 1 and self.r == 1:
            return "Both (Segment + Retrieve)"
        elif self.s == 1 and self.r == 0:
            return "Segment only"
        elif self.s == 0 and self.r == 1:
            return "Retrieve only"
        else:
            return "Idle"

def parse_clinical_intent(prompt: str, default_k: int = 5) -> ExecutionPlan:
    """Deterministic natural language instruction parser for clinical ultrasound workflows.
    
    Computes f_intent(R) -> ExecutionPlan(s, r, K).
    """
    clean = (prompt or "").strip().lower()
    
    # Check execution flags
    seg_only_keywords = ["only segment", "segment only", "just segment", "segmentation only", "no retrieval", "without search"]
    ret_only_keywords = ["only retrieve", "retrieve only", "just retrieve", "only search", "search only", "find similar only", "retrieval only", "without segmentation", "no segmentation"]
    
    if any(kw in clean for kw in seg_only_keywords):
        s = 1
        r = 0
    elif any(kw in clean for kw in ret_only_keywords):
        s = 0
        r = 1
    else:
        # Default is Both: s=1, r=1 (e.g. 'Segment and find similar cases')
        s = 1
        r = 1

    # Extract K neighbors if mentioned (e.g., 'top 3', 'k=5', '10 similar cases', 'find 5')
    k = default_k
    k_match = re.search(r'\b(?:top|k\s*=|retrieve|find)\s*(\d{1,2})\b', clean)
    if k_match:
        try:
            k = max(1, min(20, int(k_match.group(1))))
        except ValueError:
            pass

    # Extract concept
    concept = "breast lesion"
    if "cyst" in clean:
        concept = "cyst"
    elif "fibroadenoma" in clean:
        concept = "fibroadenoma"
    elif "malignant" in clean or "carcinoma" in clean:
        concept = "malignant tumor"
    elif "benign" in clean:
        concept = "benign mass"
    elif "normal" in clean:
        concept = "normal tissue"

    return ExecutionPlan(
        raw_prompt=prompt,
        s=s,
        r=r,
        k=k,
        target_concept=concept,
        filter_status="Accepted",
    )

