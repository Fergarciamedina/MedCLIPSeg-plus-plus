"""MedCLIPSeg++ : Clinical Ultrasound Retrieval & Uncertainty Triage Interface.

Designed for clinical workflow:
Step 1: Select Images (Demonstration cohort, benchmark cases, or uploaded scans)
Step 2: Analyze (Deterministic routing, selective triage, and PABP retrieval)
Step 3: Review Results (Summary metrics, case-by-case inspection, side-by-side overlays,
        opacity control, optional uncertainty view, and compact similar case cards)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import streamlit as st

# Application directory path setup
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from uncertainty_aware_retrieval import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_K,
    DEFAULT_NLIST,
    DEFAULT_NPROBE,
    FROZEN_THRESHOLD,
    EMPTY_MASK_FALLBACK_THRESHOLD,
    EmbeddingExtractor,
    IntentRouter,
    ManifestBuilder,
    PABPRetrievalEngine,
    SegmentationService,
    SelectivePredictionService,
    create_segmentation_overlay,
    load_uncertainty_visualization,
)

# Page Configuration
st.set_page_config(
    page_title="MedCLIPSeg++ | Clinical Ultrasound Retrieval",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing clinical database and indexing historical scans...")
def get_cached_system():
    """Loads and caches the 542-scan historical manifest, precomputed embeddings,

    and builds the FAISS-IVF PABP retrieval index ONCE at application launch.
    """
    builder = ManifestBuilder()
    records = builder.build_or_load_manifest()

    extractor = EmbeddingExtractor()
    embeddings = extractor.extract_or_load_gallery_embeddings(records)

    seg_service = SegmentationService(manifest=records)
    triage_service = SelectivePredictionService()
    router = IntentRouter()

    # Pre-build PABP engine over full historical gallery once
    pabp = PABPRetrievalEngine(
        nlist=DEFAULT_NLIST,
        nprobe=DEFAULT_NPROBE,
        block_size=DEFAULT_BLOCK_SIZE,
    )
    pabp.build_index(records, embeddings)

    # Fast ID lookup table for embeddings
    key_to_idx = {r["image_id"] + "@" + r["dataset"]: i for i, r in enumerate(records)}

    return {
        "builder": builder,
        "records": records,
        "embeddings": embeddings,
        "extractor": extractor,
        "seg_service": seg_service,
        "triage_service": triage_service,
        "router": router,
        "pabp": pabp,
        "key_to_idx": key_to_idx,
    }


def main():
    system = get_cached_system()
    all_records = system["records"]
    all_embeddings = system["embeddings"]
    seg_service = system["seg_service"]
    triage_service = system["triage_service"]
    router = system["router"]
    pabp_engine = system["pabp"]
    key_to_idx = system["key_to_idx"]

    # Header
    st.title("MedCLIPSeg++ : Clinical Ultrasound Retrieval & Uncertainty Triage")
    st.markdown(
        "Post-hoc quality triage and similarity search for breast ultrasound imaging "
        "using selective prediction and Predicate-Aware Block Pruning (PABP)."
    )

    # Sidebar: Workflow Configuration
    st.sidebar.header("Workflow Settings")

    retrieval_k = st.sidebar.slider(
        "Number of Similar Cases to Retrieve (K):",
        min_value=1,
        max_value=12,
        value=DEFAULT_K,
        step=1,
        help="Number of historically accepted ultrasound cases to retrieve for eligible scans.",
    )

    # Optional Research Diagnostics Expander (in sidebar, collapsed)
    with st.sidebar.expander("Research Configuration (Advanced)", expanded=False):
        diag_block_size = st.select_slider(
            "PABP Block Size (B):",
            options=[4, 8, 16],
            value=DEFAULT_BLOCK_SIZE,
            help="Size of candidate blocks in IVF inverted lists for predicate evaluation.",
        )
        diag_nprobe = st.slider(
            "IVF Probed Centroids (nprobe):",
            min_value=1,
            max_value=DEFAULT_NLIST,
            value=DEFAULT_NPROBE,
            step=1,
        )
        st.caption(
            f"Frozen Decision Threshold: tau = {FROZEN_THRESHOLD:.6f}\n"
            f"Empty Mask Fallback: tau_neg = {EMPTY_MASK_FALLBACK_THRESHOLD:.1e}"
        )

    # =========================================================================
    # STEP 1: SELECT IMAGES
    # =========================================================================
    st.subheader("Step 1: Select Images")

    mode_options = [
        "Demonstration Cohort (10 representative cases: 6 low uncertainty, 4 review required)",
        "Select from Clinical Benchmarks (BUSI, UDIAT, BUID, BUSUC, BUSBRA)",
        "Upload External Ultrasound Scans",
    ]

    # Track input mode change to reset session state analysis cleanly
    if "current_input_mode" not in st.session_state:
        st.session_state["current_input_mode"] = mode_options[0]

    input_mode = st.radio(
        "Choose an image source:",
        options=mode_options,
        index=mode_options.index(st.session_state["current_input_mode"]),
        label_visibility="collapsed",
    )

    if input_mode != st.session_state["current_input_mode"]:
        st.session_state["current_input_mode"] = input_mode
        st.session_state.pop("analysis_results", None)

    selected_queries: List[Dict[str, Any]] = []

    if "Demonstration Cohort" in input_mode:
        demo_queries, _ = system["builder"].partition_demo_queries(all_records, m=10)
        selected_queries = demo_queries
        st.markdown(
            f"**Loaded Demonstration Cohort:** `{len(selected_queries)} scans` "
            "(Stratified sample across in-domain and cross-domain clinical benchmarks)."
        )

    elif "Select from Clinical Benchmarks" in input_mode:
        col_ds, col_cases = st.columns([1, 3])
        with col_ds:
            ds_choice = st.selectbox(
                "Benchmark Cohort:",
                options=["BUSI", "UDIAT", "BUID", "BUSUC", "BUSBRA"],
                index=0,
            )
        ds_records = [r for r in all_records if r["dataset"] == ds_choice]
        with col_cases:
            chosen_ids = st.multiselect(
                "Select Cases to Analyze:",
                options=[r["image_id"] for r in ds_records],
                default=[r["image_id"] for r in ds_records[:3]],
            )
        chosen_set = set(chosen_ids)
        selected_queries = [r for r in ds_records if r["image_id"] in chosen_set]

    else:  # Upload External Ultrasound Scans
        uploaded = st.file_uploader(
            "Upload Ultrasound Scans (PNG / JPG / JPEG):",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if uploaded:
            for uf in uploaded:
                img = Image.open(uf).convert("RGB")
                selected_queries.append({
                    "image_id": uf.name,
                    "dataset": "External Upload",
                    "image_path": img,
                    "seg_mask_path": None,
                    "unc_png_path": None,
                    "unc_npy_path": None,
                    "u_fg": None,
                    "u_mean": None,
                    "count_fg": None,
                    "status": "Not Evaluated",
                    "is_unseen": True,
                })

    # Display compact thumbnails for selected scans
    if selected_queries:
        st.caption(f"Selected {len(selected_queries)} image(s) for analysis:")
        thumb_cols = st.columns(min(len(selected_queries), 8))
        for t_idx, q in enumerate(selected_queries[:8]):
            with thumb_cols[t_idx]:
                if isinstance(q["image_path"], (str, Path)):
                    t_img = Image.open(str(q["image_path"]))
                else:
                    t_img = q["image_path"]
                st.image(t_img, caption=q["image_id"][:15], use_container_width=True)
        if len(selected_queries) > 8:
            st.caption(f"... and {len(selected_queries) - 8} additional scan(s).")
    else:
        st.info("Please select or upload at least one ultrasound image to proceed.")

    st.markdown("---")

    # =========================================================================
    # STEP 2: ANALYZE
    # =========================================================================
    st.subheader("Step 2: Analyze")

    col_inst, col_btn = st.columns([3, 1])
    with col_inst:
        instruction_text = st.text_input(
            "Clinical Request / Instruction:",
            value="Segment lesions and retrieve similar historical cases",
            help="Clinical instruction parsed by the intent router (e.g. segment, retrieve, or joint workflow).",
        )
    with col_btn:
        st.write("")  # spacing
        st.write("")
        analyze_clicked = st.button("🔍 Analyze Images", type="primary", use_container_width=True)

    if analyze_clicked and selected_queries:
        parsed_instruction = router.parse(
            instruction_text,
            images=selected_queries,
            override_k=retrieval_k,
        )

        progress_bar = st.progress(0, text="Evaluating selective prediction and similarity search...")
        results_list: List[Dict[str, Any]] = []

        total_q = len(selected_queries)
        for i, q in enumerate(selected_queries):
            # 1. Segmentation lookup / adapter
            seg = seg_service.get_segmentation_for_query(
                q.get("image_path"), image_name=q["image_id"]
            )

            # 2. Selective prediction triage
            triage = triage_service.evaluate_case(
                u_fg=seg["u_fg"],
                u_mean=seg["u_mean"],
                count_fg=seg["count_fg"],
                image_id=seg["image_id"],
                dataset=seg["dataset"],
                is_unseen=seg.get("is_unseen", False),
            )

            case_result = {
                "query": q,
                "seg": seg,
                "triage": triage,
                "neighbors": [],
                "metrics": None,
            }

            # 3. Retrieval if Accepted by Policy
            if triage.status == "Accepted" and parsed_instruction.intent in ("retrieve", "segment_retrieve"):
                # Fast embedding lookup for benchmark scans; on-demand extraction for novel uploads
                q_key = q["image_id"] + "@" + q["dataset"]
                if q_key in key_to_idx:
                    q_emb = all_embeddings[key_to_idx[q_key]]
                else:
                    q_emb = system["extractor"].extract_image_embedding(q["image_path"])

                # PABP similarity search
                neighbors, metrics = pabp_engine.search_top_k(
                    query_vector=q_emb,
                    k=parsed_instruction.top_k,
                    nprobe=diag_nprobe,
                    filter_status="Accepted",
                    excluded_ids={q_key},
                )
                case_result["neighbors"] = neighbors
                case_result["metrics"] = metrics

            results_list.append(case_result)
            progress_bar.progress((i + 1) / total_q, text=f"Processed scan {i+1} of {total_q}...")

        progress_bar.empty()
        st.session_state["analysis_results"] = {
            "parsed": parsed_instruction,
            "results": results_list,
        }

    # =========================================================================
    # STEP 3: REVIEW RESULTS
    # =========================================================================
    if "analysis_results" in st.session_state:
        st.markdown("---")
        st.subheader("Step 3: Review Results")

        analysis = st.session_state["analysis_results"]
        all_cases = analysis["results"]

        accepted_cases = [c for c in all_cases if c["triage"].status == "Accepted"]
        rejected_cases = [c for c in all_cases if c["triage"].status == "Rejected"]
        unevaluated_cases = [c for c in all_cases if c["triage"].status == "Not Evaluated"]

        # 3.1 Clinical Summary Cards
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Processed Scans", f"{len(all_cases)}")
        sc2.metric("Accepted by Policy", f"{len(accepted_cases)}", help="Low predictive boundary uncertainty (U_fg <= 0.0641 or low negative entropy).")
        sc3.metric("Needs Review", f"{len(rejected_cases)}", help="Elevated predictive uncertainty (U_fg > 0.0641). Automated retrieval suppressed.")
        sc4.metric("Not Evaluated", f"{len(unevaluated_cases)}", help="Uncalibrated scans lacking dense predictive entropy maps.")

        # 3.2 Triage Tabs
        tab_accepted, tab_review = st.tabs([
            f"Eligible Cases (Accepted by Policy: {len(accepted_cases)})",
            f"Review-Required Cases (Needs Review / Unevaluated: {len(rejected_cases) + len(unevaluated_cases)})",
        ])

        # TAB 1: ELIGIBLE ACCEPTED CASES
        with tab_accepted:
            if not accepted_cases:
                st.info("No query scans met the criteria for autonomous reference retrieval in this cohort.")
            else:
                acc_labels = [f"Case #{i+1}: {c['query']['image_id']} ({c['query']['dataset']})" for i, c in enumerate(accepted_cases)]
                selected_acc_label = st.selectbox("Select case to inspect:", acc_labels, index=0)
                selected_idx = acc_labels.index(selected_acc_label)
                case_data = accepted_cases[selected_idx]

                q = case_data["query"]
                seg = case_data["seg"]
                triage = case_data["triage"]
                neighbors = case_data.get("neighbors", [])
                metrics = case_data.get("metrics")

                # Opacity control slider
                col_ctrl1, col_ctrl2 = st.columns([2, 1])
                with col_ctrl1:
                    mask_opacity = st.slider("Segmentation Overlay Opacity:", 0.0, 1.0, 0.40, 0.05, key="acc_opacity")
                with col_ctrl2:
                    show_unc_map = st.checkbox("View Predictive Uncertainty Map", value=False, key="acc_unc_toggle")

                # Side-by-side inspection
                v_cols = st.columns(2)
                if isinstance(q["image_path"], (str, Path)):
                    base_q_img = Image.open(str(q["image_path"]))
                else:
                    base_q_img = q["image_path"]

                with v_cols[0]:
                    st.markdown("**Original Ultrasound**")
                    st.image(base_q_img, use_container_width=True)

                with v_cols[1]:
                    st.markdown("**AI-Predicted Lesion Contour**")
                    overlay_img = create_segmentation_overlay(
                        base_q_img,
                        seg.get("seg_mask_path"),
                        alpha=mask_opacity,
                    )
                    st.image(overlay_img, use_container_width=True)

                # Optional Uncertainty Heatmap
                if show_unc_map:
                    st.markdown("**Dense Predictive Entropy Map (Spatial Uncertainty)**")
                    unc_img = load_uncertainty_visualization(seg.get("unc_png_path"), seg.get("unc_npy_path"))
                    if unc_img:
                        st.image(unc_img, use_container_width=True)
                    else:
                        st.info("Uncertainty map not available for this case.")

                # Clinical Triage Assessment
                st.info(
                    f"**Triage Assessment:** {triage.reason}\n\n"
                    f"*{triage.clinical_note}*"
                )

                # Retrieved Historical Cases (Top-K)
                if neighbors:
                    st.markdown(f"#### Most Similar Historical Cases (Top-{len(neighbors)} Verified Accepted Scans)")
                    n_cols = st.columns(len(neighbors))
                    for n_idx, n in enumerate(neighbors):
                        with n_cols[n_idx]:
                            n_base = Image.open(n.image_path)
                            n_ov = create_segmentation_overlay(n_base, n.seg_mask_path, alpha=0.35)
                            st.image(n_ov, use_container_width=True)
                            st.markdown(
                                f"**Match #{n_idx+1}**<br>"
                                f"Cohort: `{n.dataset}`<br>"
                                f"ID: `{n.image_id}`<br>"
                                f"Similarity: **{n.similarity*100:.1f}%**",
                                unsafe_allow_html=True,
                            )

                # Collapsed Research Diagnostics
                with st.expander("🔬 Research Diagnostics (PABP Profiling & Numerical Verification)", expanded=False):
                    if metrics:
                        st.markdown("**PABP Algorithmic Metrics (Detailed Breakdown):**")
                        dm1, dm2, dm3 = st.columns(3)
                        dm1.metric("IVF Inverted Lists Probed", f"{metrics.lists_probed}")
                        dm2.metric("Total Blocks Inspected", f"{metrics.blocks_inspected}")
                        dm3.metric("Blocks Pruned by PABP", f"{metrics.blocks_pruned_pabp}", f"{metrics.block_pruning_ratio*100:.1f}% of blocks")

                        dm4, dm5, dm6 = st.columns(3)
                        dm4.metric("Candidates in Pruned Blocks", f"{metrics.candidates_in_pruned_blocks}", help="Saved strictly via block-level pruning before computing vector dot products.")
                        dm5.metric("Candidates Filtered in Scanned Blocks", f"{metrics.candidates_filtered_in_scanned_blocks}", help="Ineligible candidates skipped inside retained blocks.")
                        dm6.metric("Vector Distances Computed", f"{metrics.vector_distances_computed}", help="Exact inner product calculations performed.")

                        st.caption(
                            f"Total candidates in probed lists: `{metrics.total_candidates_in_probed_lists}` | "
                            f"Candidates excluded by Query ID: `{metrics.candidates_excluded_by_id}` | "
                            f"Overall Candidate Distance Savings: `{metrics.total_candidates_avoided} / {metrics.total_candidates_in_probed_lists}` ({metrics.overall_candidate_saving_ratio*100:.1f}%)"
                        )
                    if seg.get("dsc") is not None:
                        st.markdown(
                            f"**Ground-Truth Benchmark Reference (Research Validation Only):** "
                            f"Dice Coefficient = `{seg['dsc']:.4f}` | Normalized Surface Distance = `{seg.get('nsd', 0.0):.4f}`"
                        )

        # TAB 2: REVIEW REQUIRED CASES
        with tab_review:
            flagged_cases = rejected_cases + unevaluated_cases
            if not flagged_cases:
                st.success("No cases flagged for manual review in this cohort.")
            else:
                rev_labels = [f"Review #{i+1}: {c['query']['image_id']} ({c['query']['dataset']}) — {c['triage'].status}" for i, c in enumerate(flagged_cases)]
                selected_rev_label = st.selectbox("Select case requiring review:", rev_labels, index=0)
                selected_rev_idx = rev_labels.index(selected_rev_label)
                rev_data = flagged_cases[selected_rev_idx]

                rq = rev_data["query"]
                rseg = rev_data["seg"]
                rtriage = rev_data["triage"]

                col_rctrl1, col_rctrl2 = st.columns([2, 1])
                with col_rctrl1:
                    rev_opacity = st.slider("Segmentation Overlay Opacity:", 0.0, 1.0, 0.40, 0.05, key="rev_opacity")
                with col_rctrl2:
                    show_r_unc = st.checkbox("View Predictive Uncertainty Map", value=True, key="rev_unc_toggle")

                rv_cols = st.columns(2)
                if isinstance(rq["image_path"], (str, Path)):
                    base_rq_img = Image.open(str(rq["image_path"]))
                else:
                    base_rq_img = rq["image_path"]

                with rv_cols[0]:
                    st.markdown("**Original Ultrasound**")
                    st.image(base_rq_img, use_container_width=True)

                with rv_cols[1]:
                    st.markdown("**AI-Predicted Lesion Contour (Needs Review)**")
                    roverlay_img = create_segmentation_overlay(
                        base_rq_img,
                        rseg.get("seg_mask_path"),
                        alpha=rev_opacity,
                        color=(255, 69, 0),  # Red-orange indicator for review cases
                    )
                    st.image(roverlay_img, use_container_width=True)

                if show_r_unc:
                    st.markdown("**Dense Predictive Entropy Map (Spatial Uncertainty)**")
                    r_unc_img = load_uncertainty_visualization(rseg.get("unc_png_path"), rseg.get("unc_npy_path"))
                    if r_unc_img:
                        st.image(r_unc_img, use_container_width=True)
                    else:
                        st.info("Uncertainty map not available for this case.")

                st.warning(
                    f"**Triage Reason:** {rtriage.reason}\n\n"
                    f"*{rtriage.clinical_note}*"
                )

                with st.expander("🔬 Research Diagnostics", expanded=False):
                    if rseg.get("dsc") is not None:
                        st.markdown(
                            f"**Ground-Truth Reference (Research Validation Only):** "
                            f"Dice Coefficient = `{rseg['dsc']:.4f}` | Normalized Surface Distance = `{rseg.get('nsd', 0.0):.4f}`"
                        )
                    st.caption("Automated similarity retrieval was suppressed in accordance with the selective prediction policy.")

    # Clinical Disclaimer (subtle footer)
    st.markdown("---")
    st.caption(
        "Clinical Prototype Disclaimer: Developed for Concordia University ELEC 6661 research demonstration. "
        "Intended for investigational analysis and quality triage; not approved as a medical device."
    )


if __name__ == "__main__":
    main()
