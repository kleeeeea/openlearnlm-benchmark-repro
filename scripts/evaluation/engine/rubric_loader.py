"""Rubric Loader for Scenario-Specific Long Answer Evaluation

Loads English evaluation rubrics from JSON file for LLM-as-Judge evaluation.
Rubrics are pre-translated from the original Korean CSV.

JSON Source: scripts/evaluation/rubrics_translated.json
Original CSV: 2. Literature Review/LLM 역할표(1231)_1231.csv
"""

import json
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RubricInfo:
    """Rubric information for a sub-scenario"""
    sub_scenario_en: str
    sub_scenario_ko: str
    scenario_en: str
    scenario_ko: str
    rubric_en: str
    rubric_ko: str


class RubricLoader:
    """Load and manage scenario-specific rubrics for Long Answer evaluation"""

    # Relative path from this file's directory (scripts/evaluation/engine/)
    RUBRICS_JSON_PATH = "../rubrics_translated.json"

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize RubricLoader.

        Args:
            base_path: Base path for relative file lookups.
                       Defaults to the directory containing this file.
        """
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))

        self.base_path = base_path
        self._rubrics: Dict[str, RubricInfo] = {}
        self._loaded = False

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve relative path from base path"""
        return os.path.normpath(os.path.join(self.base_path, relative_path))

    def load(self) -> None:
        """Load rubrics from JSON file"""
        if self._loaded:
            return

        json_path = self._resolve_path(self.RUBRICS_JSON_PATH)

        with open(json_path, 'r', encoding='utf-8') as f:
            rubrics_data = json.load(f)

        for sub_scenario_en, data in rubrics_data.items():
            # Only include rubrics that have English translation
            if data.get('rubric_en'):
                self._rubrics[sub_scenario_en] = RubricInfo(
                    sub_scenario_en=data.get('sub_scenario_en', sub_scenario_en),
                    sub_scenario_ko=data.get('sub_scenario_ko', ''),
                    scenario_en=data.get('scenario_en', ''),
                    scenario_ko=data.get('scenario_ko', ''),
                    rubric_en=data.get('rubric_en', ''),
                    rubric_ko=data.get('rubric_ko', '')
                )

        self._loaded = True

    @staticmethod
    def parse_scenario_field(scenario_field: str) -> Tuple[str, str]:
        """
        Parse scenario field to extract scenario and sub-scenario.

        Args:
            scenario_field: e.g., "Learner Analysis / Individual Learner Level Assessment"

        Returns:
            Tuple of (scenario, sub_scenario)
        """
        if '/' not in scenario_field:
            return scenario_field.strip(), ""

        parts = scenario_field.split('/', 1)
        scenario = parts[0].strip()
        sub_scenario = parts[1].strip() if len(parts) > 1 else ""

        return scenario, sub_scenario

    def get_rubric(self, sub_scenario_en: str) -> Optional[RubricInfo]:
        """
        Get rubric for an English sub-scenario.

        Args:
            sub_scenario_en: English sub-scenario name

        Returns:
            RubricInfo if found, None otherwise
        """
        if not self._loaded:
            self.load()

        return self._rubrics.get(sub_scenario_en.strip())

    def get_rubric_text(self, scenario_field: str) -> Tuple[str, str]:
        """
        Get English rubric text for a scenario field from test data.

        Args:
            scenario_field: Full scenario field, e.g.,
                           "Learner Analysis / Individual Learner Level Assessment"

        Returns:
            Tuple of (rubric_text_en, rubric_source)
            rubric_source is "scenario_specific" or "generic"
        """
        if not self._loaded:
            self.load()

        _, sub_scenario_en = self.parse_scenario_field(scenario_field)

        if not sub_scenario_en:
            return "", "generic"

        rubric_info = self.get_rubric(sub_scenario_en)

        if rubric_info and rubric_info.rubric_en:
            return rubric_info.rubric_en, "scenario_specific"

        return "", "generic"

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded rubrics"""
        if not self._loaded:
            self.load()

        return {
            "total_rubrics": len(self._rubrics),
            "rubrics_with_en": sum(1 for r in self._rubrics.values() if r.rubric_en),
            "rubrics_with_ko": sum(1 for r in self._rubrics.values() if r.rubric_ko),
        }


# Singleton instance for reuse
_loader_instance: Optional[RubricLoader] = None


def get_rubric_loader() -> RubricLoader:
    """Get singleton RubricLoader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = RubricLoader()
    return _loader_instance
