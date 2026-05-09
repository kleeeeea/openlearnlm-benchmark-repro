# OpenLearnLM Benchmark

A comprehensive benchmark for evaluating Large Language Models (LLMs) on educational tasks, designed for teacher education and pedagogical contexts.

## Overview

OpenLearnLM Benchmark evaluates LLMs across four dimensions critical for educational applications:

| Dimension | Description | Items |
|-----------|-------------|-------|
| **Skills** | Scenario-based educational interaction abilities | - |
| **Content Knowledge** | Subject matter expertise across disciplines | - |
| **Pedagogical Knowledge** | Teaching methodology and educational theory | 1,143 |
| **Attitude** | Epistemological, instructional, and ethical stances | 14 |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/openlearnlm-benchmark.git
cd openlearnlm-benchmark

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

### Running Evaluation

```bash
# Evaluate a single model on all benchmarks
python scripts/evaluation/run_evaluation.py --model gpt-4 --benchmark all

# Evaluate on specific benchmark
python scripts/evaluation/run_evaluation.py --model gpt-4 --benchmark pedagogical_knowledge
```

### Generating Reports

```bash
# Generate integrated evaluation report
python scripts/evaluation/generate_integrated_report.py --input evaluation_responses/
```

## Data Format

All benchmark data is stored in JSONL format with the following structure:

```json
{
  "question": "Question text",
  "options": ["A", "B", "C", "D"],
  "answer": "B",
  "item_id": "unique_identifier",
  "metadata": {
    "subject": "Education",
    "difficulty": "medium",
    "domain": "cognitive",
    "question_type": "multiple_choice",
    "language": "en",
    "source": "chile"
  }
}
```

## Directory Structure

```
openlearnlm-benchmark/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── scripts/
│   ├── evaluation/      # Model evaluation scripts
│   └── generation/      # Question generation scripts
└── data/
    ├── skills/                  # Functional skills benchmark (split files)
    ├── content_knowledge/       # Subject content benchmark
    ├── pedagogical_knowledge/   # Teaching methodology benchmark
    └── attitude/                # Professional disposition benchmark
```

## Data Preparation

### Skills Data (Large Files)

The skills training data is split into multiple parts due to file size limitations. Merge them before use:

```bash
# Merge split training files
cat data/skills/train_part_*.jsonl > data/skills/questions_train.jsonl

# Verify the merged file
wc -l data/skills/questions_train.jsonl
# Expected: approximately 120,000 lines
```

The test file (`questions_test.jsonl`) is provided as a single file and requires no merging.

## Benchmark Details

### Pedagogical Knowledge

Based on the pedagogy-benchmark dataset (Lelievre et al., 2025), covering:
- Instructional Strategies
- Learning Theories
- Classroom Management
- Special Education
- Assessment Methods

### Attitude Benchmark

Evaluates LLM attitudes across:
- **Epistemological stance** (Items 1-4): Knowledge and learning beliefs
- **Instructional stance** (Items 5-8): Teaching approach preferences
- **Normative & ethical stance** (Items 9-12): Professional ethics
- **Deception detection** (Items 13-14): Alignment faking methodology

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{openlearnlm2025,
  title={OpenLearnLM: A Benchmark for Evaluating Educational Large Language Models},
  author={OpenLearnLM Team},
  year={2025},
  url={https://github.com/your-repo/openlearnlm-benchmark}
}
```

## Data Sources and Licenses

- **Pedagogical Knowledge**: Apache 2.0 (from [pedagogy-benchmark](https://huggingface.co/datasets/AI-for-Education/pedagogy-benchmark))
- **Attitude Benchmark**: Original contribution

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- EduBench team for benchmark design inspiration
- Lelievre et al. for the pedagogy-benchmark dataset
- lm-evaluation-harness for evaluation framework reference
