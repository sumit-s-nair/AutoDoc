# Data Format Specification

This document defines the training data format for fine-tuning the AutoDoc model.

## Final Training Format

We use the **Alpaca instruction-tuning format** for compatibility with most fine-tuning frameworks.

### Schema

```json
{
  "instruction": "string - What the model should do",
  "input": "string - Code context to document", 
  "output": "string - Expected documentation"
}
```

### Example Entry

```json
{
  "instruction": "Generate a concise README for this Python project based on its structure and main files.",
  "input": "Project: fastapi-users
            Structure:
            ├── fastapi_users/
            │   ├── __init__.py
            │   ├── authentication/
            │   ├── router/
            │   └── manager.py
            ├── tests/
            ├── pyproject.toml
            └── README.md

            Main module (manager.py):
            ```python
            class UserManager:
                ""Manages user operations.""
                async def create(self, user_create):
                    ...
            ```",
  "output": "# FastAPI Users
            Ready-to-use user authentication for FastAPI applications.
            ## Features
            - User registration and login
            - JWT and cookie authentication
            - Customizable user manager
            
            ## Installation
            ```bash
            pip install fastapi-users
            ```\n
            
            ## Quick Start
            ```python
            from fastapi_users import UserManager
            ```"  
}
```

---

## Input Format Variants

### Variant A: Full Repository Context

For README generation from entire repos:

```
Project: {repo_name}

Structure:
{file_tree}

Key Files:
{file_name_1}:
```{language}
{file_content}
```,

{file_name_2}:
...
```

### Variant B: Single File Documentation

For function/class documentation:

```
Language: {language}
File: {filename}

```{language}
{code_content}
```,
```

### Variant C: Diff-Based Updates

For updating docs after code changes:

```
Original Code:
```{language}
{old_code}
```,

Updated Code:
```{language}
{new_code}
```,

Current Documentation:
{current_docs}
```

---

## Output Guidelines

Documentation output should follow these principles:

### ✅ Good Output Characteristics
- Starts with a one-line description
- Uses bullet points for features
- Includes minimal, working code examples
- No filler phrases ("This is a...", "The purpose of...")

### ❌ Avoid
- "This function does..." (obvious)
- Restating parameter names as descriptions
- Placeholder text
- Overly verbose explanations

---

## File Naming Convention

Processed data files follow this naming:

```
data/processed/{source}_{split}_{version}.jsonl
```

Examples:
- `codesearchnet_train_v1.jsonl`
- `github_readme_train_v1.jsonl`
- `combined_val_v1.jsonl`

---

## Validation Schema

Each entry is validated against:

```python
SCHEMA = {
    "instruction": {
        "type": "string",
        "min_length": 10,
        "max_length": 500
    },
    "input": {
        "type": "string",
        "min_length": 50,
        "max_length": 8000
    },
    "output": {
        "type": "string",
        "min_length": 20,
        "max_length": 4000
    }
}
```

---

## Statistics Tracking

For each processed dataset, we track:

| Metric | Description |
|--------|-------------|
| `total_entries` | Number of examples |
| `avg_input_tokens` | Average input length |
| `avg_output_tokens` | Average output length |
| `language_distribution` | % per programming language |
| `instruction_types` | Distribution of task types |
