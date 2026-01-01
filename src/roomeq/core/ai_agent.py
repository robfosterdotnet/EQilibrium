"""AI Agent for explaining room analysis and EQ corrections.

Uses Azure OpenAI to generate detailed educational explanations
of room measurements and recommended corrections.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AzureOpenAI

from roomeq.core.analysis import AnalysisResult, RoomProblem
from roomeq.core.eq_optimizer import OptimizationResult

# Load environment variables
load_dotenv()


@dataclass
class AnalysisExplanation:
    """Container for AI-generated explanation."""

    summary: str
    detailed_analysis: str
    eq_explanation: str
    learning_points: str
    full_text: str


def get_azure_client() -> AzureOpenAI:
    """Create Azure OpenAI client from environment variables."""
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def format_problems_for_prompt(problems: list[RoomProblem]) -> str:
    """Format room problems for the AI prompt."""
    if not problems:
        return "No significant problems detected."

    lines = []
    for p in problems:
        problem_type = "Peak (boost)" if p.is_peak else "Dip (cut)"
        lines.append(
            f"- {problem_type} at {p.frequency:.0f} Hz: {p.magnitude:+.1f} dB, "
            f"Q={p.q_factor:.1f}, Severity: {p.severity.value}"
        )
    return "\n".join(lines)


def format_eq_bands_for_prompt(optimization: OptimizationResult | None) -> str:
    """Format EQ bands for the AI prompt."""
    if optimization is None or not optimization.settings.bands:
        return "No EQ corrections generated."

    lines = []
    for i, band in enumerate(optimization.settings.bands, 1):
        lines.append(
            f"- Band {i}: {band.frequency:.0f} Hz, Gain: {band.gain:+.1f} dB, Q: {band.q:.1f}"
        )
    return "\n".join(lines)


def generate_analysis_prompt(
    left_analysis: AnalysisResult | None,
    right_analysis: AnalysisResult | None,
    left_optimization: OptimizationResult | None,
    right_optimization: OptimizationResult | None,
) -> str:
    """Generate the prompt for the AI explanation."""
    # Gather statistics
    left_rms = left_analysis.rms_deviation if left_analysis else 0
    right_rms = right_analysis.rms_deviation if right_analysis else 0
    left_problems = left_analysis.problems if left_analysis else []
    right_problems = right_analysis.problems if right_analysis else []

    # Calculate improvement
    left_improvement = (
        left_optimization.improvement_db if left_optimization else 0
    )
    right_improvement = (
        right_optimization.improvement_db if right_optimization else 0
    )

    prompt = f"""You are an expert audio engineer and acoustics educator. A user has just completed
room acoustic measurements using RoomEQ software. Please provide a detailed, educational
explanation of their results and the recommended corrections.

## Measurement Results

### Left Channel
- RMS Deviation from flat: {left_rms:.1f} dB
- Detected Problems:
{format_problems_for_prompt(left_problems)}

### Right Channel
- RMS Deviation from flat: {right_rms:.1f} dB
- Detected Problems:
{format_problems_for_prompt(right_problems)}

## Recommended EQ Corrections

### Left Channel EQ
{format_eq_bands_for_prompt(left_optimization)}
- Estimated improvement: {left_improvement:.1f} dB RMS reduction

### Right Channel EQ
{format_eq_bands_for_prompt(right_optimization)}
- Estimated improvement: {right_improvement:.1f} dB RMS reduction

---

Please provide a comprehensive explanation covering:

1. **Summary** (2-3 sentences): What's the overall state of this room's acoustics?

2. **Detailed Analysis**:
   - Explain what each detected problem means acoustically
   - What physical phenomena cause peaks and dips at these frequencies?
   - Are there any patterns that suggest specific room issues (room modes, reflections, etc.)?
   - How do the left and right channels compare?

3. **EQ Corrections Explained**:
   - For each significant EQ band, explain WHY this correction is being applied
   - What is Q factor and why does each band have its specific Q value?
   - Explain any trade-offs or limitations of EQ correction

4. **Learning Points**:
   - What are "room modes" and how do they relate to the detected peaks?
   - Why can't EQ fully fix dips/nulls?
   - Tips for acoustic treatment that could complement EQ
   - What frequencies are most critical for music listening?

Please use clear, educational language suitable for someone learning about room acoustics.
Use specific numbers from the measurements to make your explanation concrete and relevant."""

    return prompt


def explain_analysis(
    left_analysis: AnalysisResult | None,
    right_analysis: AnalysisResult | None,
    left_optimization: OptimizationResult | None,
    right_optimization: OptimizationResult | None,
) -> AnalysisExplanation:
    """
    Generate an AI-powered explanation of room analysis and corrections.

    Args:
        left_analysis: Analysis result for left channel
        right_analysis: Analysis result for right channel
        left_optimization: EQ optimization for left channel
        right_optimization: EQ optimization for right channel

    Returns:
        AnalysisExplanation with detailed educational content
    """
    client = get_azure_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")

    prompt = generate_analysis_prompt(
        left_analysis, right_analysis, left_optimization, right_optimization
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert audio engineer and acoustics educator. "
                    "You explain complex acoustic concepts in clear, accessible terms "
                    "while maintaining technical accuracy. You use the specific measurement "
                    "data provided to give concrete, actionable explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    full_text = response.choices[0].message.content or ""

    # Parse sections from the response
    sections = {
        "summary": "",
        "detailed_analysis": "",
        "eq_explanation": "",
        "learning_points": "",
    }

    current_section = None
    current_content: list[str] = []

    for line in full_text.split("\n"):
        line_lower = line.lower()
        if "summary" in line_lower and line.startswith("#"):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = "summary"
            current_content = []
        elif "detailed analysis" in line_lower and line.startswith("#"):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = "detailed_analysis"
            current_content = []
        elif "eq correction" in line_lower and line.startswith("#"):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = "eq_explanation"
            current_content = []
        elif "learning" in line_lower and line.startswith("#"):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = "learning_points"
            current_content = []
        elif current_section:
            current_content.append(line)

    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return AnalysisExplanation(
        summary=sections["summary"] or full_text[:500],
        detailed_analysis=sections["detailed_analysis"] or "",
        eq_explanation=sections["eq_explanation"] or "",
        learning_points=sections["learning_points"] or "",
        full_text=full_text,
    )


def explain_analysis_sync(
    left_analysis: AnalysisResult | None,
    right_analysis: AnalysisResult | None,
    left_optimization: OptimizationResult | None,
    right_optimization: OptimizationResult | None,
) -> str:
    """
    Synchronous version that returns just the full text explanation.

    Useful for simpler integration where section parsing isn't needed.
    """
    explanation = explain_analysis(
        left_analysis, right_analysis, left_optimization, right_optimization
    )
    return explanation.full_text
