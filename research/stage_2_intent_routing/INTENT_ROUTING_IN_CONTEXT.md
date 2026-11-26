# Stage 2 · Where intent routing fits

Location `research/stage_2_intent_routing/` · [Up to research/](../RESEARCH_GUIDE.md)

The overall system accepts a web request describing what the user wants to see, such as a segmentation overlay, similar reference cases, or both. The application uses a rule-based [intent router](../../application/uncertainty_aware_retrieval/uncertainty_aware_retrieval/intent_router.py) with a small set of actions. The parsed operation and requested result count are passed downstream. This stage keeps the request interface predictable.

Stage 4 compares scores on saved masks and entropy maps generated with fixed offline lesion descriptions. User requests and inferred intents are application inputs. The evaluation concerns the uncertainty scores and the frozen review rule. [Study input types](../stage_1_study_inputs/STUDY_INPUTS.md#which-inputs-were-used) · [Application routing settings and software tests](../../application/stages/stage2_intent_router/INTENT_ROUTER.md).

## Credit and related work

Stage 2 reuses an external contribution whose original author and source are pending verification. Keyword-triggered dispatch and the separation of intent from arguments have precedents in the language-processing literature. The application guide relates those papers and regular-expression matching to the rules actually used. [Weizenbaum (1966)](https://doi.org/10.1145/365153.365168) · [Bocklisch et al. (2017)](https://arxiv.org/abs/1712.05181) · [Application credit](../../application/stages/stage2_intent_router/INTENT_ROUTER.md#implementation-credit) · [Design basis](../../application/stages/stage2_intent_router/INTENT_ROUTER.md#research-background-and-design-basis).

[Research overview](../RESEARCH_GUIDE.md) · [Next · Segmentation artifacts](../stage_3_medclipseg/FROZEN_SEGMENTATION.md)
