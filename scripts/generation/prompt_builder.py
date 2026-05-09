"""Prompt Builder for Question Generation"""

from dataclasses import dataclass
from typing import List, Dict

from korean_to_english_mapping import KoreanToEnglishTranslator


@dataclass
class RowMetadata:
    """Metadata extracted from CSV row"""
    center: str
    role: str
    scenario: str
    sub_scenario: str
    subject: str
    difficulty: str
    domain: str
    bloom_level: str
    question_type: str


class PromptBuilder:
    """Build prompts from CSV row metadata"""

    SYSTEM_PROMPT = """You are an expert educational assessment designer.
Your task is to generate high-quality educational questions based on the given context and specifications.

CRITICAL RULES:
1. Output MUST be in valid JSON format with exactly these fields: "question", "answer"
2. For multiple choice questions, the "answer" field MUST contain ONLY a single letter: A, B, C, or D
   - CORRECT: "answer": "B"
   - WRONG: "answer": "B) Blue square"
   - WRONG: "answer": "The answer is B because..."
3. The question MUST directly serve the specified scenario's purpose
4. Do NOT create generic knowledge questions unrelated to the scenario
5. STRICT MCQ FORMAT RULE: For multiple choice questions:
   - NEVER use these phrases: "and why", "explain why", "justify your choice", "describe why", "tell why"
   - NEVER ask for explanations or justifications in MCQ format
   - GOOD: "Which of the following best represents...? A) ... B) ... C) ... D) ..."
   - BAD: "Which do you prefer and why? A) ..."
   - BAD: "Select the best option and justify your choice."
   - Simply ask "Which...", "What...", or "Select..." with 4 options. No explanation requests.
6. ANSWER ACCURACY RULE: Before finalizing the answer:
   - Verify that the chosen answer is LOGICALLY and FACTUALLY correct
   - Ensure only ONE option is clearly the best answer (no ambiguity)
   - If multiple options could be correct, redesign the question or options
   - Double-check: Does the correct answer truly match the question's intent?
   - Avoid questions where the "correct" answer is debatable or subjective"""

    QUESTION_TEMPLATE = """Generate an educational question based on the following specifications:

## Context
- **Educational Center**: {center}
- **User Role**: {role}
- **Scenario**: {scenario}
- **Sub-Scenario**: {sub_scenario}

### Scenario Definition
{scenario_definition}

### Sub-Scenario Definition
{sub_scenario_definition}

## Subject & Difficulty
- **Subject Area**: {subject}
- **Difficulty Level**: {difficulty} (Bloom's level: {bloom_level})
- **Learning Domain**: {domain}

### Difficulty Guidelines
{difficulty_guidance}

## Question Type
- **Format**: {question_type}

## Requirements
1. The question MUST directly serve the scenario purpose described above
2. For "{domain}" domain at "{bloom_level}" level:
   - {domain_guidance}
3. Question format guidelines:
   - {format_guidance}

## Output Format (JSON only, no markdown)
{{"question": "The complete question text including any answer choices if multiple choice", "answer": "{answer_format}"}}

Generate the question in English."""

    DOMAIN_GUIDANCE = {
        ("cognitive", "Remembering"): "Focus on recall of facts, terms, and basic concepts",
        ("cognitive", "Understanding"): "Focus on explaining ideas and concepts, interpreting meaning",
        ("cognitive", "Applying"): "Focus on using information in new situations",
        ("cognitive", "Analyzing"): "Focus on breaking down information, finding patterns and relationships",
        ("cognitive", "Evaluating"): "Focus on justifying decisions, critiquing, and making judgments",
        ("cognitive", "Creating"): "Focus on producing new or original work, designing solutions",
        ("affective", "Receiving"): "Focus on awareness and willingness to hear",
        ("affective", "Responding"): "Focus on active participation and reaction",
        ("affective", "Valuing"): "Focus on attaching worth and expressing preferences",
        ("affective", "Organizing"): "Focus on prioritizing values and resolving conflicts",
        ("affective", "Characterizing(internalizing values)"): "Focus on consistent behavior reflecting internalized values",
    }

    FORMAT_GUIDANCE = {
        "multiple choice": """Provide 4 answer options (A, B, C, D) with one correct answer.
IMPORTANT:
- The answer MUST be a single letter only (A, B, C, or D)
- Do NOT include "and why", "explain", "justify", or any request for explanation
- Format: "Which/What/Select... [question]? A) option1 B) option2 C) option3 D) option4"
- Each option should be self-contained without needing explanation""",
        "long answer": """Design an open-ended question requiring detailed explanation or analysis.
IMPORTANT for the answer field:
- Provide an actual MODEL ANSWER that directly answers the question
- Write as if a knowledgeable student is answering the question
- Do NOT write grading rubrics like "Students should..." or "The answer should include..."
- Do NOT write evaluation criteria or scoring guidelines
- Write a concrete, complete answer example""",
    }

    DIFFICULTY_GUIDANCE = {
        "Easy": "Use Bloom's lower levels (Remembering, Understanding). Focus on: recall of facts, defining terms, explaining concepts, summarizing information. Use verbs like: list, define, describe, explain, summarize.",
        "Medium": "Use Bloom's middle levels (Applying, Analyzing). Focus on: using knowledge in new situations, comparing/contrasting, identifying patterns, breaking down information. Use verbs like: apply, demonstrate, analyze, compare, differentiate.",
        "Hard": "Use Bloom's higher levels (Evaluating, Creating). Focus on: making judgments, critiquing, designing new solutions, synthesizing information. Use verbs like: evaluate, justify, create, design, formulate.",
    }

    ANSWER_FORMAT = {
        "multiple choice": "SINGLE LETTER ONLY (A, B, C, or D)",
        "long answer": "A complete model answer written as if answering the question directly (NOT a rubric or grading criteria)",
    }

    def __init__(self, strict_mode: bool = True):
        """
        Initialize PromptBuilder with translator.

        Args:
            strict_mode: If True, raises error when mapping not found.
        """
        self.translator = KoreanToEnglishTranslator(strict_mode=strict_mode)

    def build_prompt(self, metadata: RowMetadata) -> List[Dict[str, str]]:
        """Build API messages from metadata"""
        # Translate Korean to English
        translation = self.translator.translate_metadata(
            center=metadata.center,
            scenario=metadata.scenario,
            sub_scenario=metadata.sub_scenario
        )

        domain_key = (metadata.domain.lower().strip(), metadata.bloom_level.strip())
        domain_guidance = self.DOMAIN_GUIDANCE.get(
            domain_key,
            "Ensure the question matches the specified cognitive/affective level"
        )

        format_guidance = self.FORMAT_GUIDANCE.get(
            metadata.question_type.lower().strip(),
            "Provide a clear, well-structured question"
        )

        # Get difficulty guidance
        difficulty_guidance = self.DIFFICULTY_GUIDANCE.get(
            metadata.difficulty,
            "Ensure the question complexity matches the specified difficulty level."
        )

        # Get answer format
        answer_format = self.ANSWER_FORMAT.get(
            metadata.question_type.lower().strip(),
            "The correct answer"
        )

        # Format definitions with fallback
        scenario_def = translation.scenario_definition or f"Educational scenario: {translation.scenario_en}"
        sub_scenario_def = translation.sub_scenario_definition or f"Sub-scenario: {translation.sub_scenario_en}"

        user_prompt = self.QUESTION_TEMPLATE.format(
            center=translation.center_en,
            role=metadata.role,
            scenario=translation.scenario_en,
            sub_scenario=translation.sub_scenario_en,
            scenario_definition=scenario_def,
            sub_scenario_definition=sub_scenario_def,
            subject=metadata.subject,
            difficulty=metadata.difficulty,
            difficulty_guidance=difficulty_guidance,
            domain=metadata.domain,
            bloom_level=metadata.bloom_level,
            question_type=metadata.question_type,
            domain_guidance=domain_guidance,
            format_guidance=format_guidance,
            answer_format=answer_format
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
