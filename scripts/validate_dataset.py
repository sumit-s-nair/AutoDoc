"""
AutoDoc Dataset Validator
=========================
Validates processed datasets for format correctness and quality.

Usage:
    python validate_dataset.py [--input data/processed] [--stats] [--sample N]
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import jsonschema


# Expected schema for training data
TRAINING_SCHEMA = {
    "type": "object",
    "required": ["instruction", "input", "output"],
    "properties": {
        "instruction": {
            "type": "string",
            "minLength": 10,
            "maxLength": 500
        },
        "input": {
            "type": "string",
            "minLength": 20,
            "maxLength": 10000
        },
        "output": {
            "type": "string",
            "minLength": 10,
            "maxLength": 5000
        },
        "source": {"type": "string"},
        "language": {"type": "string"},
    }
}


def validate_entry(entry: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a single entry against the schema.
    
    Args:
        entry: Data entry to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        jsonschema.validate(entry, TRAINING_SCHEMA)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)


def validate_file(file_path: Path) -> dict:
    """
    Validate all entries in a JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        Validation results
    """
    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": []
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            results["total"] += 1
            
            try:
                entry = json.loads(line)
                is_valid, error = validate_entry(entry)
                
                if is_valid:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    if len(results["errors"]) < 10:  # Limit error collection
                        results["errors"].append({
                            "line": line_num,
                            "error": error
                        })
            except json.JSONDecodeError as e:
                results["invalid"] += 1
                results["errors"].append({
                    "line": line_num,
                    "error": f"JSON parse error: {e}"
                })
    
    return results


def compute_stats(file_path: Path) -> dict:
    """
    Compute statistics for a dataset file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        Statistics dictionary
    """
    stats = {
        "count": 0,
        "input_lengths": [],
        "output_lengths": [],
        "sources": {},
        "languages": {},
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            stats["count"] += 1
            stats["input_lengths"].append(len(entry["input"]))
            stats["output_lengths"].append(len(entry["output"]))
            
            source = entry.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            lang = entry.get("language", "unknown")
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
    
    # Compute aggregates
    if stats["input_lengths"]:
        stats["avg_input_length"] = sum(stats["input_lengths"]) / len(stats["input_lengths"])
        stats["avg_output_length"] = sum(stats["output_lengths"]) / len(stats["output_lengths"])
        stats["min_input_length"] = min(stats["input_lengths"])
        stats["max_input_length"] = max(stats["input_lengths"])
    
    # Remove raw lists to save memory
    del stats["input_lengths"]
    del stats["output_lengths"]
    
    return stats


def sample_entries(file_path: Path, n: int = 5) -> list:
    """
    Sample random entries from a dataset file.
    
    Args:
        file_path: Path to JSONL file
        n: Number of samples
    
    Returns:
        List of sampled entries
    """
    import random
    
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    
    if len(entries) <= n:
        return entries
    
    return random.sample(entries, n)


def main():
    parser = argparse.ArgumentParser(
        description="Validate AutoDoc processed datasets"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed",
        help="Input directory with processed data"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Show N sample entries"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    print("=" * 50)
    print("AutoDoc Dataset Validator")
    print("=" * 50)
    print(f"Input: {input_dir.absolute()}")
    
    if not input_dir.exists():
        print(f"\n⚠ Directory not found: {input_dir}")
        return
    
    # Find dataset files
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"\n⚠ No .jsonl files found in {input_dir}")
        return
    
    all_valid = True
    
    for file_path in jsonl_files:
        print(f"\n📄 {file_path.name}")
        print("-" * 40)
        
        # Validate
        results = validate_file(file_path)
        
        print(f"  Total entries: {results['total']:,}")
        print(f"  Valid: {results['valid']:,}")
        print(f"  Invalid: {results['invalid']:,}")
        
        if results["invalid"] > 0:
            all_valid = False
            print(f"\n  ⚠ Sample errors:")
            for err in results["errors"][:3]:
                print(f"    Line {err['line']}: {err['error'][:60]}...")
        
        # Stats
        if args.stats and results["valid"] > 0:
            stats = compute_stats(file_path)
            print(f"\n  📊 Statistics:")
            print(f"    Avg input length: {stats['avg_input_length']:.0f} chars")
            print(f"    Avg output length: {stats['avg_output_length']:.0f} chars")
            print(f"\n    Sources:")
            for source, count in stats["sources"].items():
                print(f"      {source}: {count:,}")
            print(f"\n    Top languages:")
            top_langs = sorted(stats["languages"].items(), key=lambda x: -x[1])[:5]
            for lang, count in top_langs:
                print(f"      {lang}: {count:,}")
        
        # Samples
        if args.sample > 0:
            print(f"\n  📝 Sample entries:")
            samples = sample_entries(file_path, args.sample)
            for i, entry in enumerate(samples, 1):
                print(f"\n    --- Sample {i} ---")
                print(f"    Instruction: {entry['instruction'][:60]}...")
                print(f"    Input: {entry['input'][:100]}...")
                print(f"    Output: {entry['output'][:100]}...")
    
    # Summary
    print("\n" + "=" * 50)
    if all_valid:
        print("✅ All datasets validated successfully!")
    else:
        print("⚠ Some validation errors found. Review above.")


if __name__ == "__main__":
    main()
