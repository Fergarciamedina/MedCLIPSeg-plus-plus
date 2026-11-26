# Stage 2 · Route the user's request

Location `application/stages/stage2_intent_router/` · [Up to stages/](../APPLICATION_STAGES.md)

The [application router](../../uncertainty_aware_retrieval/uncertainty_aware_retrieval/intent_router.py) takes the user's instruction and returns a supported intent, a requested result count, and an explanation. The action set is small, so fixed keyword and number rules give repeatable results and can be exercised without a separate language model.

## Implementation credit

Stage 2 incorporates an existing rule-based router from an external contributor. The original author and source are pending verification. The project history records the import of the web router and notebook parser in the same commit. [Repository import record](https://github.com/Fergarciamedina/MedCLIPSeg-plus-plus/commit/74a9dfca5f3587a2effd12b0a261af8d3601bf44).

The dashboard backend calls the runtime router. The six-stage notebook calls this folder's [intent_router.py](intent_router.py), which produces separate segmentation and retrieval flags and clamps a recognized count to 1–20. [__init__.py](__init__.py) exposes the parser and its execution-plan record. The web text box uses the runtime settings below. [Dashboard backend](../../medical_dashboard/backend/API_BACKEND.md).

## Research background and design basis

The runtime parser lowercases the instruction, checks two keyword sets, and extracts a result count using a regular expression. The keyword sets yield two Boolean signals for segmentation and retrieval. Their combination selects the action. Extracting `K` separately lets the same retrieval action support different result counts. For a small command vocabulary, these rules make dispatch repeatable and avoid a second model or an inference service. Substring rules have limited coverage for unseen wording and negation. [Python `re` documentation](https://docs.python.org/3/library/re.html).

These papers provide methodological context for the code used here. The external implementation credit remains open.

| Decision in Stage 2 | Scientific or computational precedent |
| --- | --- |
| Recognize supported operations through keywords. | Weizenbaum shows how keyword-triggered rules can dispatch actions over constrained text. [*ELIZA* (1966)](https://doi.org/10.1145/365153.365168). |
| Extract an explicit integer from forms such as `top 3` or `k=3`. | Thompson describes a computational method for matching regular expressions in text. The local parser uses Python's `re` API for its numeric patterns. [*Regular Expression Search Algorithm* (1968)](https://doi.org/10.1145/363347.363387). |
| Return an operation and its argument separately. | Bocklisch et al. describe the separation of intent recognition and entity extraction in task-oriented language understanding. The local operation and `K` serve those roles with fixed rules. [*Rasa: Open Source Language Understanding and Dialogue Management* (2017)](https://arxiv.org/abs/1712.05181). |

## Runtime configuration and examples

| Example request | Route |
| --- | --- |
| `Segment the lesion` | Segmentation view with the matched case's stored result. |
| `Find 3 similar cases` | Retrieval request with `K=3`, subject to Stage 4 acceptance. |
| `Segment and show 5 similar cases` | Combined segmentation and retrieval request. |

The default limit is **5** reference matches. The router extracts explicit integers from **1 to 50** in supported patterns. It flags an empty or unrecognized request as ambiguous and falls back to the combined workflow. The backend still applies the same triage condition before retrieving cases. A matched case can expose its stored result for any route. Request text selects the operation; it does not rerun the segmentation model or change the mask.

Five regression tests cover segmentation, retrieval, combined intent, ambiguity and a custom result count. A held-out evaluation of paraphrases and routing accuracy is outside the study scope. Stage 4 uses masks and entropy maps generated with fixed offline lesion descriptions. [Router tests](../../uncertainty_aware_retrieval/tests/test_intent_router.py) · [Study inputs](../../../research/stage_1_study_inputs/STUDY_INPUTS.md#which-inputs-were-used).

[Application stage map](../APPLICATION_STAGES.md) · [Next · Segmentation artifacts](../stage3_segmentation_backbone/SEGMENTATION_ARTIFACTS.md)
