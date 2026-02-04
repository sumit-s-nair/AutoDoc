"""
AutoDoc Data Preprocessor
=========================
Cleans and formats raw datasets into instruction-tuning format.

Pipeline steps:
1. Load raw data (parquet, json)
2. Clean and filter entries
3. Convert to Alpaca instruction format
4. Create train/val/test splits
5. Save as JSONL

Usage:
    python preprocess.py [--input data/raw] [--output data/processed]
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Generator

import pandas as pd
from tqdm import tqdm


# Diverse instruction templates for better generalization
INSTRUCTION_TEMPLATES = {
    "function_doc": [
        # Direct commands
        "Generate documentation for this function.",
        "Write a concise docstring for this code.",
        "Document this function with its parameters and return value.",
        "Write a brief description of what this code does.",
        "Create documentation for the following code.",
        "Add a docstring to this function.",
        "Write the documentation for this code snippet.",
        # Questions
        "What does this function do?",
        "Can you explain this code?",
        "What is the purpose of this function?",
        "Describe what this code accomplishes.",
        # Specific requests
        "Summarize this function in one sentence.",
        "Write a one-line description for this function.",
        "Explain the inputs and outputs of this function.",
        "Document this code following best practices.",
        "Write a clear and concise docstring.",
        # Contextual
        "As a developer, write documentation for this code.",
        "Help me understand what this function does.",
        "Generate a helpful docstring for other developers.",
        "Write documentation that explains the purpose and usage.",
        # Short but valid (10+ chars)
        "Docstring for this code:",
        "Document this code:",
        "Explain this code briefly:",
        "What does this code do?",
        # Detailed style
        "Write comprehensive documentation including parameters, return values, and any exceptions.",
        "Create a detailed docstring with examples if applicable.",
    ],
    "readme_gen": [
        "Generate a README for this project based on its structure.",
        "Write documentation for this codebase.",
        "Create a concise project README from this code structure.",
        "Write a project overview based on this file structure.",
        "Generate project documentation.",
    ],
    "code_explain": [
        "Explain what this code does concisely.",
        "Summarize the purpose of this code.",
        "Describe this code's functionality.",
        "What is this code doing?",
        "Break down this code for me.",
        "Explain the logic in this code.",
    ],
}


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    
    return text.strip()


def is_quality_doc(doc: str) -> bool:
    """Check if documentation meets quality standards."""
    if not doc or len(doc) < 10:
        return False
    
    # Filter out low-quality patterns
    low_quality_patterns = [
        r"^TODO",
        r"^FIXME",
        r"^XXX",
        r"^\.\.\.$",
        r"^Add description",
        r"^No description",
        r"^Undocumented",
        r"^@param",
        r"^@return",
    ]
    
    for pattern in low_quality_patterns:
        if re.match(pattern, doc.strip(), re.IGNORECASE):
            return False
    
    # Require some substance (not just a single word)
    if len(doc.split()) < 3:
        return False
    
    return True


def process_codesearchnet(raw_dir: Path) -> Generator[dict, None, None]:
    """
    Process CodeSearchNet dataset.
    Files are stored as: codesearchnet/{language}.parquet
    """
    csn_dir = raw_dir / "codesearchnet"
    
    if not csn_dir.exists():
        return
    
    # Process each language parquet file
    for parquet_file in csn_dir.glob("*.parquet"):
        language = parquet_file.stem  # e.g., "python" from "python.parquet"
        
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"  ⚠ Error reading {parquet_file}: {e}")
            continue
        
        for _, row in df.iterrows():
            # Get code and docstring (field names from our download script)
            code = row.get("code", "")
            doc = row.get("docstring", "")
            
            # Skip short code or poor quality docs
            if not code or len(code.strip()) < 20 or not is_quality_doc(doc):
                continue
            
            # Clean the documentation
            doc = clean_text(doc)
            
            # Format code with language marker
            code_input = f"```{language}\n{code.strip()}\n```"
            
            yield {
                "instruction": random.choice(INSTRUCTION_TEMPLATES["function_doc"]),
                "input": code_input,
                "output": doc,
                "source": "codesearchnet",
                "language": language,
            }


def process_code_docstring_corpus(raw_dir: Path) -> Generator[dict, None, None]:
    """
    Process Code Docstring Corpus dataset.
    Files are stored as: code_docstring_corpus/train.parquet
    """
    corpus_dir = raw_dir / "code_docstring_corpus"
    
    if not corpus_dir.exists():
        return
    
    for parquet_file in corpus_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"  ⚠ Error reading {parquet_file}: {e}")
            continue
        
        for _, row in df.iterrows():
            # Columns are: desc (description), decl (declaration), bodies (code)
            code = row.get("bodies", "")
            doc = row.get("desc", "")
            decl = row.get("decl", "")
            
            if not code or len(code.strip()) < 20 or not is_quality_doc(doc):
                continue
            
            doc = clean_text(doc)
            
            # Include declaration if available for context
            if decl:
                code_input = f"```python\n{decl.strip()}\n{code.strip()}\n```"
            else:
                code_input = f"```python\n{code.strip()}\n```"
            
            yield {
                "instruction": random.choice(INSTRUCTION_TEMPLATES["function_doc"]),
                "input": code_input,
                "output": doc,
                "source": "code_docstring_corpus",
                "language": "python",
            }


def process_python_instructions(raw_dir: Path) -> Generator[dict, None, None]:
    """
    Process Python Code Instructions dataset (already in Alpaca format).
    Files are stored as: python_code_instructions/train.parquet
    """
    instr_dir = raw_dir / "python_code_instructions"
    
    if not instr_dir.exists():
        return
    
    for parquet_file in instr_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"  ⚠ Error reading {parquet_file}: {e}")
            continue
        
        for _, row in df.iterrows():
            instruction = row.get("instruction", row.get("prompt", ""))
            input_text = row.get("input", "")
            output_text = row.get("output", row.get("response", ""))
            
            if not instruction or not output_text:
                continue
            
            yield {
                "instruction": clean_text(instruction),
                "input": clean_text(input_text),
                "output": clean_text(output_text),
                "source": "python_code_instructions",
                "language": "python",
            }


def process_the_stack_smol(raw_dir: Path) -> Generator[dict, None, None]:
    """
    Process The Stack Smol dataset (whole-file code with docs).
    Files are stored as: the_stack_smol/{language}.parquet
    """
    stack_dir = raw_dir / "the_stack_smol"
    
    if not stack_dir.exists():
        return
    
    for parquet_file in stack_dir.glob("*.parquet"):
        language = parquet_file.stem
        
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"  ⚠ Error reading {parquet_file}: {e}")
            continue
        
        for _, row in df.iterrows():
            content = row.get("content", "")
            
            if not content or len(content) < 100:
                continue
            
            # Extract first docstring as documentation target
            code_input = f"```{language}\n{content[:2000].strip()}\n```"
            
            yield {
                "instruction": random.choice(INSTRUCTION_TEMPLATES["code_explain"]),
                "input": code_input,
                "output": "Analyze the code structure and provide documentation.",
                "source": "the_stack_smol",
                "language": language,
            }


def create_splits(
    examples: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """Create train/val/test splits."""
    random.seed(seed)
    random.shuffle(examples)
    
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return (
        examples[:train_end],
        examples[train_end:val_end],
        examples[val_end:]
    )


def save_jsonl(examples: list, output_path: Path):
    """Save examples as JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for AutoDoc fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory with raw data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits"
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Max examples per source (0 = unlimited)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only process a sample and show stats"
    )
    
    args = parser.parse_args()
    
    raw_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("AutoDoc Data Preprocessor")
    print("=" * 50)
    print(f"Input: {raw_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    
    # Collect all examples with progress bars
    all_examples = []
    sources = {}
    max_per = args.max_per_source if args.max_per_source > 0 else float('inf')
    dry_run_limit = 100 if args.dry_run else float('inf')
    
    # Process each data source
    processors = [
        ("CodeSearchNet", process_codesearchnet),
        ("Code Docstring Corpus", process_code_docstring_corpus),
        ("Python Instructions", process_python_instructions),
        ("The Stack Smol", process_the_stack_smol),
    ]
    
    for name, processor in processors:
        print(f"\n📦 Processing {name}...")
        sources[name] = 0
        
        for example in tqdm(processor(raw_dir), desc=name):
            all_examples.append(example)
            sources[name] += 1
            
            if sources[name] >= min(max_per, dry_run_limit):
                break
    
    if not all_examples:
        print("\n⚠ No data found to process!")
        print("\nExpected data structure:")
        print("  data/raw/codesearchnet/*.parquet")
        print("  data/raw/code_docstring_corpus/*.parquet")
        print("  data/raw/python_code_instructions/*.parquet")
        return
    
    # Create splits
    print("\n🔀 Creating train/val/test splits...")
    train, val, test = create_splits(
        all_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Save
    if not args.dry_run:
        save_jsonl(train, output_dir / "train.jsonl")
        save_jsonl(val, output_dir / "val.jsonl")
        save_jsonl(test, output_dir / "test.jsonl")
    
    # Metadata
    metadata = {
        "total_examples": len(all_examples),
        "sources": sources,
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "seed": args.seed,
    }
    
    if not args.dry_run:
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 50)
    print("Preprocessing Summary")
    print("=" * 50)
    print(f"  Total examples: {len(all_examples):,}")
    print(f"\n  By source:")
    for source, count in sources.items():
        if count > 0:
            print(f"    {source}: {count:,}")
    print(f"\n  Splits:")
    print(f"    Train: {len(train):,}")
    print(f"    Val: {len(val):,}")
    print(f"    Test: {len(test):,}")
    
    if args.dry_run:
        print("\n  [DRY RUN - no files saved]")
        print("\n  Sample entry:")
        sample = json.dumps(all_examples[0], indent=2, ensure_ascii=False)
        print(sample[:600] + "..." if len(sample) > 600 else sample)
    else:
        print(f"\n  Output files:")
        print(f"    {output_dir / 'train.jsonl'}")
        print(f"    {output_dir / 'val.jsonl'}")
        print(f"    {output_dir / 'test.jsonl'}")
        print(f"    {output_dir / 'metadata.json'}")
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
