"""Shared prompt templates for all experiments."""

# ============================================================================
# System Prompt Components
# ============================================================================

SYSTEM_PROMPT_BASE = "You are an expert academic reviewer tasked with evaluating research papers."

TASK_INSTRUCTION = """
I am giving you a paper. I want to predict its acceptance outcome at ICLR.
Note: ICLR generally has a ~30% acceptance rate."""

BOXED_FORMAT_INSTRUCTION = """
Your answer must start with: \\boxed{Accept} or \\boxed{Reject}"""

JSON_FORMAT_INSTRUCTION = """
Respond with a reasoning trace starting with a strictly formatted JSON block.

Provide the review in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{
  "summary": "string",
  "strengths": "string",
  "weaknesses": "string",
  "score": integer,  // Score 1-10
  "decision": "Accept" OR "Reject"
}
```"""


# ============================================================================
# Role Modifiers (B2: Role Prompts)
# ============================================================================

CRITICAL_MODIFIER = """
IMPORTANT: Be critical of claims and reject papers that lack substance. Look for methodological flaws, unsupported claims, and insufficient experimental validation. When in doubt, reject."""

ENTHUSIASTIC_MODIFIER = """
IMPORTANT: Be less critical of claims and accept papers that are on the border. Focus on the potential contribution and novelty. Give papers the benefit of the doubt when the core idea is sound."""

STANDARD_MODIFIER = ""  # No modifier — baseline


# ============================================================================
# Base Model Prompt (H1: completion-style for non-instruct models)
# ============================================================================

BASE_COMPLETION_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Read the following paper and predict whether it was accepted or rejected at ICLR. "
    "ICLR generally has a ~30% acceptance rate. "
    "Your answer must start with: \\boxed{Accept} or \\boxed{Reject}"
)


# ============================================================================
# Meta-review Prompt (B1: PDR, B2: Strategy D)
# ============================================================================

METAREVIEW_SYSTEM_PROMPT = """You are an Area Chair at a prestigious machine learning conference. You are in charge of meta-reviewing a paper that was reviewed by 5 reviewers. Your job is to aggregate the reviews into a single meta-review in the same format. Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."""

METAREVIEW_USER_TEMPLATE = """Review 1/5: {review1}

Review 2/5: {review2}

Review 3/5: {review3}

Review 4/5: {review4}

Review 5/5: {review5}

Respond with a reasoning trace followed by a strictly formatted JSON block.

1. First, provide your reasoning under the section "THOUGHT:".
2. Then, provide the metareview in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{{
  "metareview": "string",
  "soundness": integer, // Score 1-5
  "presentation": integer, // Score 1-5
  "contribution": integer, // Score 1-5
  "overall": integer, // Score 1-10
  "confidence": integer, // Score 1-5
  "decision": "accept" OR "reject"
}}
```"""


# ============================================================================
# Strategy D: Multi-Perspective Meta-Review (B2)
# ============================================================================

STRATEGY_D_SYSTEM_PROMPT = """You are an Area Chair at a prestigious machine learning conference. You have received two reviews of the same paper: one from a critical reviewer who focused on finding weaknesses, and one from an enthusiastic reviewer who focused on finding strengths. Your job is to synthesize both perspectives and make a final accept/reject decision."""

STRATEGY_D_USER_TEMPLATE = """CRITICAL REVIEW (focused on weaknesses):
{critical_review}

ENTHUSIASTIC REVIEW (focused on strengths):
{enthusiastic_review}

Synthesize both perspectives. Consider whether the strengths outweigh the weaknesses.

Respond with a reasoning trace followed by a strictly formatted JSON block.

1. First, provide your reasoning under the section "THOUGHT:".
2. Then, provide the decision in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{{
  "synthesis": "string",
  "soundness": integer, // Score 1-5
  "presentation": integer, // Score 1-5
  "contribution": integer, // Score 1-5
  "overall": integer, // Score 1-10
  "confidence": integer, // Score 1-5
  "decision": "Accept" OR "Reject"
}}
```"""


# ============================================================================
# Strategy D Critical: More critical meta-review variant (B2)
# ============================================================================

STRATEGY_D_CRITICAL_SYSTEM_PROMPT = """You are an Area Chair at a prestigious machine learning conference with high standards. You have received two reviews of the same paper: one from a critical reviewer who focused on finding weaknesses, and one from an enthusiastic reviewer who focused on finding strengths. Your job is to synthesize both perspectives and make a rigorous final accept/reject decision."""

STRATEGY_D_CRITICAL_USER_TEMPLATE = """CRITICAL REVIEW (focused on weaknesses):
{critical_review}

ENTHUSIASTIC REVIEW (focused on strengths):
{enthusiastic_review}

IMPORTANT: Carefully weigh whether the weaknesses outweigh the strengths. Consider the severity of methodological issues, validity of claims, and rigor of experimental validation. When it is unclear whether the paper meets the bar, REJECT. Only accept papers where strengths clearly outweigh weaknesses.

Respond with a reasoning trace followed by a strictly formatted JSON block.

1. First, provide your reasoning under the section "THOUGHT:".
2. Then, provide the decision in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{{
  "synthesis": "string",
  "soundness": integer, // Score 1-5
  "presentation": integer, // Score 1-5
  "contribution": integer, // Score 1-5
  "overall": integer, // Score 1-10
  "confidence": integer, // Score 1-5
  "decision": "accept" OR "reject"
}}
```"""


# ============================================================================
# Contamination Check Prompts (H2)
# ============================================================================

TITLE_ONLY_SYSTEM_PROMPT = (
    "You are a knowledgeable AI research assistant. You will be given only the title of "
    "an academic paper. Your task is to generate the abstract for this paper."
)

TITLE_ONLY_USER_TEMPLATE = (
    "Given the following paper title, generate the abstract for this paper. "
    "Write only the abstract text, nothing else.\n\n"
    "Title: {title}"
)

CONTENT_ABLATION_SYSTEM_PROMPT = (
    "You are an expert academic reviewer tasked with evaluating research papers.\n"
    "I am giving you a paper (or part of a paper). I want to predict its acceptance outcome at ICLR.\n"
    "Note: ICLR generally has a ~30% acceptance rate.\n"
    "Your answer must start with: \\boxed{Accept} or \\boxed{Reject}"
)


# ============================================================================
# Helper: Build system prompt from components
# ============================================================================

def build_system_prompt(modifier=None, output_format="boxed"):
    """Build a complete system prompt.

    Args:
        modifier: None, "critical", "enthusiastic", or "standard"
        output_format: "boxed" or "json"
    """
    parts = [SYSTEM_PROMPT_BASE, TASK_INSTRUCTION]

    if modifier == "critical":
        parts.append(CRITICAL_MODIFIER)
    elif modifier == "enthusiastic":
        parts.append(ENTHUSIASTIC_MODIFIER)
    # "standard" or None = no modifier

    if output_format == "boxed":
        parts.append(BOXED_FORMAT_INSTRUCTION)
    else:
        parts.append(JSON_FORMAT_INSTRUCTION)

    return "\n".join(parts)
