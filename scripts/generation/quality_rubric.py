"""Quality Assessment Rubric for LLM-as-Judge Evaluation"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RubricCriteria:
    """Single evaluation criterion"""
    name: str
    description: str
    weight: float = 1.0
    critical: bool = False  # If critical, low score causes automatic FAIL


# MCQ Evaluation Criteria
MCQ_CRITERIA = [
    RubricCriteria(
        name="answer_accuracy",
        description="Is the marked correct answer logically and factually correct?",
        weight=1.0,
        critical=True  # Score < 3 = FAIL
    ),
    RubricCriteria(
        name="question_clarity",
        description="Is the question clear, unambiguous, and grammatically correct?",
        weight=1.0,
        critical=True  # Score < 2 = FAIL
    ),
    RubricCriteria(
        name="distractor_quality",
        description="Are the wrong options plausible but clearly distinguishable from the correct answer?",
        weight=1.0,
        critical=False
    ),
    RubricCriteria(
        name="difficulty_match",
        description="Does the question match the specified Bloom's taxonomy level?",
        weight=1.0,
        critical=False
    ),
    RubricCriteria(
        name="scenario_alignment",
        description="Does the question fit the educational scenario and context?",
        weight=1.0,
        critical=False
    ),
]

# Long Answer Additional Criteria
LONG_ANSWER_CRITERIA = [
    RubricCriteria(
        name="answer_completeness",
        description="Does the model answer fully address the question with sufficient depth?",
        weight=1.0,
        critical=False
    ),
]


JUDGE_SYSTEM_PROMPT_MCQ = """You are an expert educational assessment evaluator specializing in multiple-choice question quality.

Your task is to evaluate the given MCQ based on these 5 criteria (score 1-5 each):

## Evaluation Criteria

1. **Answer Accuracy** (CRITICAL)
   - Is the marked answer (A, B, C, or D) logically and factually correct?
   - Would educational experts agree this is the best answer?
   - Score 1-2: Wrong or highly debatable answer
   - Score 3: Acceptable but with minor issues
   - Score 4-5: Clearly correct answer

2. **Question Clarity**
   - Is the question stem clear and unambiguous?
   - Is it grammatically correct and well-structured?
   - Score 1-2: Confusing or poorly written
   - Score 3: Understandable but could be clearer
   - Score 4-5: Clear and precise

3. **Distractor Quality**
   - Are wrong options (distractors) plausible but distinguishable?
   - Do they represent common misconceptions or reasonable alternatives?
   - Score 1-2: Obvious wrong answers or too similar to correct answer
   - Score 3: Acceptable distractors
   - Score 4-5: Well-designed distractors

4. **Difficulty Match**
   - Does the cognitive demand match the specified Bloom's level?
   - Easy = Remembering/Understanding, Medium = Applying/Analyzing, Hard = Evaluating/Creating
   - For Affective domain: Easy = Receiving/Responding, Medium = Valuing/Organizing, Hard = Characterizing
   - Score 1-2: Significant mismatch
   - Score 3: Roughly appropriate
   - Score 4-5: Perfect match

5. **Scenario Alignment**
   - Does the question serve the educational scenario's purpose?
   - Is it relevant to the specified subject and context?
   - Score 1-2: Irrelevant or misaligned
   - Score 3: Loosely connected
   - Score 4-5: Directly serves the scenario

## Output Format (JSON only)
{
  "scores": {
    "answer_accuracy": <1-5>,
    "question_clarity": <1-5>,
    "distractor_quality": <1-5>,
    "difficulty_match": <1-5>,
    "scenario_alignment": <1-5>
  },
  "total_score": <5-25>,
  "issues": ["list of specific issues found, if any"],
  "pass": <true if total >= 20 AND answer_accuracy >= 3 AND question_clarity >= 2, else false>
}"""


JUDGE_SYSTEM_PROMPT_LONG_ANSWER = """You are an expert educational assessment evaluator specializing in open-ended question quality.

Your task is to evaluate the given long-answer question based on these 6 criteria (score 1-5 each):

## Evaluation Criteria

1. **Answer Accuracy** (CRITICAL)
   - Is the model answer factually and conceptually correct?
   - Score 1-2: Contains significant errors
   - Score 3: Mostly correct with minor issues
   - Score 4-5: Accurate and reliable

2. **Question Clarity**
   - Is the question clear about what is being asked?
   - Score 1-2: Vague or confusing
   - Score 3: Understandable but could be clearer
   - Score 4-5: Clear and well-defined

3. **Answer Completeness**
   - Does the model answer fully address the question?
   - Score 1-2: Incomplete or superficial
   - Score 3: Addresses main points
   - Score 4-5: Comprehensive and thorough

4. **Difficulty Match**
   - Does the question match the specified Bloom's level?
   - Score 1-2: Significant mismatch
   - Score 3: Roughly appropriate
   - Score 4-5: Perfect match

5. **Scenario Alignment**
   - Does the question serve the educational scenario's purpose?
   - Score 1-2: Irrelevant
   - Score 3: Loosely connected
   - Score 4-5: Directly serves the scenario

6. **Pedagogical Value**
   - Does the question promote meaningful learning?
   - Score 1-2: Low educational value
   - Score 3: Standard value
   - Score 4-5: High educational value

## Output Format (JSON only)
{
  "scores": {
    "answer_accuracy": <1-5>,
    "question_clarity": <1-5>,
    "answer_completeness": <1-5>,
    "difficulty_match": <1-5>,
    "scenario_alignment": <1-5>,
    "pedagogical_value": <1-5>
  },
  "total_score": <6-30>,
  "issues": ["list of specific issues found, if any"],
  "pass": <true if total >= 24 AND answer_accuracy >= 3, else false>
}"""


def build_judge_prompt(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build evaluation prompt for a single item"""
    metadata = item.get("metadata", {})
    question_type = metadata.get("question_type", "multiple choice").lower().strip()

    # Select appropriate system prompt
    if question_type == "long answer":
        system_prompt = JUDGE_SYSTEM_PROMPT_LONG_ANSWER
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT_MCQ

    # Build user prompt with item details
    user_prompt = f"""## Item to Evaluate

**Question Type**: {metadata.get("question_type", "N/A")}
**Subject**: {metadata.get("subject", "N/A")}
**Difficulty**: {metadata.get("difficulty", "N/A")}
**Domain**: {metadata.get("domain", "N/A")}
**Scenario**: {metadata.get("scenario", "N/A")}

### Question
{item.get("question", "")}

### Marked Answer
{item.get("answer", "")}

---
Evaluate this item and provide your assessment in JSON format."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# Pass/Fail thresholds
MCQ_PASS_THRESHOLD = 20  # out of 25
MCQ_CRITICAL_THRESHOLDS = {
    "answer_accuracy": 3,  # minimum score to pass
    "question_clarity": 2,  # minimum score to pass
}

LONG_ANSWER_PASS_THRESHOLD = 24  # out of 30
LONG_ANSWER_CRITICAL_THRESHOLDS = {
    "answer_accuracy": 3,
}
