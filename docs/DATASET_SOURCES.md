# Dataset Sources

This document details all dataset sources used for fine-tuning the AutoDoc model.

## Tier 1: Pre-Built Datasets

Ready-to-use, high-quality datasets from the ML community.

### CodeSearchNet

| Property | Value |
|----------|-------|
| **Size** | ~2M code-docstring pairs |
| **Languages** | Python, JavaScript, Java, Go, Ruby, PHP |
| **Source** | Open-source GitHub repositories |
| **Access** | [HuggingFace](https://huggingface.co/datasets/code-search-net/code_search_net) |

**What it contains:**
- Function-level code snippets
- Associated docstrings/comments

**Usage:**
```bash
python scripts/download_datasets.py --datasets codesearchnet --languages python,javascript --max-examples 50000
```

---

### Code Docstring Corpus

| Property | Value |
|----------|-------|
| **Size** | ~148K code-docstring pairs |
| **Languages** | Python |
| **Source** | GitHub repositories |
| **Access** | [HuggingFace](https://huggingface.co/datasets/teven/code_docstring_corpus) |

**What it contains:**
- Python code at top-level and class-level
- Associated docstrings

**Usage:**
```bash
python scripts/download_datasets.py --datasets docstring --max-examples 50000
```

---

### Python Code Instructions (Alpaca format)

| Property | Value |
|----------|-------|
| **Size** | ~18K instruction-output pairs |
| **Format** | Alpaca instruction-tuning format |
| **Source** | Curated Python coding tasks |
| **Access** | [HuggingFace](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) |

**What it contains:**
- Instructions like "Write a function that..."
- Complete code solutions

**Usage:**
```bash
python scripts/download_datasets.py --datasets instructions --max-examples 18000
```

---

### The Stack Smol

| Property | Value |
|----------|-------|
| **Size** | Subset of The Stack |
| **Languages** | Multiple programming languages |
| **Source** | BigCode permissively licensed |
| **Access** | [HuggingFace](https://huggingface.co/datasets/bigcode/the-stack-smol) |

**What it contains:**
- Real code files with documentation
- Filtered for files containing docstrings

**Usage:**
```bash
python scripts/download_datasets.py --datasets stack-smol --languages python,javascript
```

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
