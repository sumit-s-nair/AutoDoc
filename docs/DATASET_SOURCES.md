# Dataset Sources

This document details all dataset sources used for fine-tuning the AutoDoc model.

## Tier 1: Pre-Built Datasets

Ready-to-use, high-quality datasets from the ML community.

### CodeSearchNet

| Property | Value |
|----------|-------|
| **Size** | ~2 million (code, comment) pairs |
| **Languages** | Python, JavaScript, Java, Go, Ruby, PHP |
| **Source** | Open-source GitHub repositories |
| **License** | Various (MIT, Apache, etc.) |
| **Access** | [HuggingFace](https://huggingface.co/datasets/code_search_net) |

**What it contains:**
- Function-level code snippets
- Associated docstrings/comments
- Natural language descriptions

**Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("code_search_net", "python")
```

---

### The Vault

| Property | Value |
|----------|-------|
| **Size** | Large-scale multilingual |
| **Languages** | 10+ programming languages |
| **Source** | The Stack (permissively licensed) |
| **License** | Permissive licenses only |
| **Access** | [HuggingFace](https://huggingface.co/datasets/Fsoft-AIC/the-vault-function) |

**What it contains:**
- Function-level code with documentation
- High-quality filtering applied
- Deduplicated entries

**Usage:**
```python
from datasets import load_dataset
dataset = load_dataset("Fsoft-AIC/the-vault-function", split="train")
```

---

### CoDocBench

| Property | Value |
|----------|-------|
| **Size** | ~10K pairs |
| **Focus** | Code-documentation alignment |
| **Source** | Academic research |
| **License** | Research use |
| **Paper** | [arXiv:2407.02630](https://arxiv.org/abs/2407.02630) |

**What it contains:**
- Code changes paired with documentation updates
- Tracks how docs evolve with code
- High alignment quality

---

## Tier 2: Custom Scraping

Custom data collection for README-focused documentation.

### GitHub README Scraping

**Target repositories:**
- Repos with 500+ stars
- Active in last 2 years
- Has README.md > 500 characters
- English language

**Data extracted:**
- `README.md` content
- Repository structure (file tree)
- Package/config files (package.json, pyproject.toml, etc.)
- Key source files (entry points)

**Filtering criteria:**
- Exclude template-generated READMEs
- Exclude READMEs with < 3 sections
- Exclude non-English content
- Exclude auto-generated docs

---

### Awesome Lists

Curated collections of well-documented projects.

**Sources:**
- `awesome-python`
- `awesome-javascript`
- `awesome-go`
- `awesome-rust`

**Why these:**
- Community-vetted quality
- Consistently good documentation
- Diverse project types

---

## Data Quality Indicators

We prioritize datasets based on:

| Indicator | Weight | Description |
|-----------|--------|-------------|
| **Alignment** | High | Code and docs must be semantically related |
| **Conciseness** | High | Prefer terse, useful docs over verbose ones |
| **Recency** | Medium | Modern coding practices preferred |
| **Diversity** | Medium | Multiple languages and domains |
| **License** | High | Must be permissively licensed |

---

## License Compliance

All datasets used comply with:
- Original repository licenses
- Fair use for research/training
- No copyleft restrictions on derived models

We maintain a license audit log in `data/licenses/` for all scraped content.
