"""Stage 1: Deterministic Natural-Language Intent Router.

Parses clinical instructions to route user requests into:
1. 'segment': MedCLIPSeg segmentation & uncertainty quantification only.
2. 'retrieve': Visual similarity search only (bypassing new segmentation).
3. 'segment_retrieve': Combined segmentation, selective triage, and conditioned retrieval.
"""

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .config import DEFAULT_K


@dataclass
class ParsedInstruction:
    raw_instruction: str
    intent: str  # 'segment', 'retrieve', 'segment_retrieve'
    image_count: int
    top_k: int
    is_ambiguous: bool
    explanation: str


class IntentRouter:
    """Lightweight, deterministic rule-based natural-language intent parser."""

    SEGMENT_KEYWORDS = [
        "segment",
        "segmentation",
        "outline",
        "delineate",
        "mask",
        "contour",
        "lesion boundary",
        "detect lesion",
        "delineation",
    ]

    RETRIEVE_KEYWORDS = [
        "retrieve",
        "retrieval",
        "similar",
        "similarity",
        "find cases",
        "nearest",
        "search",
        "neighbors",
        "match",
        "look for similar",
        "historical cases",
    ]

    def __init__(self, default_k: int = DEFAULT_K):
        self.default_k = default_k

    def parse(
        self,
        instruction: str,
        images: Optional[List[Any]] = None,
        override_k: Optional[int] = None,
    ) -> ParsedInstruction:
        """Parse natural language instruction and determine application actions."""
        raw_text = (instruction or "").strip().lower()
        image_count = len(images) if images is not None else 0

        # Extract K if mentioned in instruction (e.g. "top 3", "k=10", "5 similar")
        k_val = override_k or self._extract_k(raw_text) or self.default_k

        # Detect intent flags
        has_segment = any(kw in raw_text for kw in self.SEGMENT_KEYWORDS)
        has_retrieve = any(kw in raw_text for kw in self.RETRIEVE_KEYWORDS)

        is_ambiguous = False
        explanation = ""

        if has_segment and has_retrieve:
            intent = "segment_retrieve"
            explanation = "Detected joint intent: segmenting query scans, evaluating selective uncertainty triage, and retrieving similar historical cases."
        elif has_segment:
            intent = "segment"
            explanation = "Detected segmentation intent: performing MedCLIPSeg segmentation and uncertainty quantification."
        elif has_retrieve:
            intent = "retrieve"
            explanation = f"Detected similarity retrieval intent: searching for Top-{k_val} similar historical cases."
        else:
            # Ambiguous or empty fallback: default to segment_retrieve
            intent = "segment_retrieve"
            is_ambiguous = True
            explanation = (
                f"Instruction was ambiguous or empty ('{instruction}'). "
                "Defaulting to combined workflow: Segment + Retrieve with selective triage."
            )

        return ParsedInstruction(
            raw_instruction=instruction,
            intent=intent,
            image_count=image_count,
            top_k=k_val,
            is_ambiguous=is_ambiguous,
            explanation=explanation,
        )

    def _extract_k(self, text: str) -> Optional[int]:
        """Extract requested top-K value if explicitly specified."""
        patterns = [
            r"\btop[- ]?(\d+)\b",
            r"\b(\d+)\s+(?:most\s+)?similar\b",
            r"\bk\s*=\s*(\d+)\b",
            r"\bfind\s+(\d+)\b",
            r"\bshow\s+(\d+)\b",
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                try:
                    val = int(match.group(1))
                    if 1 <= val <= 50:
                        return val
                except ValueError:
                    pass
        return None

