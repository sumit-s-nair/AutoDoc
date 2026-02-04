"""
AutoDoc Dataset Downloader
==========================
Downloads pre-built datasets from HuggingFace for fine-tuning.

Supported datasets:
- codesearchnet: ~2M (code, docstring) pairs across 6 languages
- code_docstring_corpus: Python code with docstrings
- bigcode/the-stack-smol: Small subset of The Stack

Usage:
    python download_datasets.py [--datasets codesearchnet] [--output data/raw]
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def download_codesearchnet(output_dir: Path, languages: list = None, max_per_lang: int = 50000) -> dict:
    """
    Download CodeSearchNet dataset directly from HuggingFace ZIP files.
    
    Args:
        output_dir: Directory to save the dataset
        languages: List of languages to download
        max_per_lang: Maximum examples per language
    
    Returns:
        Statistics about downloaded data
    """
    languages = languages or ["python", "javascript", "java", "go"]
    stats = {"total_examples": 0, "languages": {}}
    
    print(f"\n📦 Downloading CodeSearchNet for: {languages}")
    
    csn_dir = output_dir / "codesearchnet"
    csn_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/data"
    
    for lang in languages:
        try:
            print(f"  Downloading {lang}...")
            
            # Download ZIP file
            url = f"{base_url}/{lang}.zip"
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            # Get total size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            data = io.BytesIO()
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=lang) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    data.write(chunk)
                    pbar.update(len(chunk))
            
            data.seek(0)
            
            # Extract and parse JSONL.GZ files from ZIP
            import gzip
            examples = []
            with zipfile.ZipFile(data) as zf:
                # Find all .jsonl.gz files in training data
                train_files = [n for n in zf.namelist() if 'train' in n and n.endswith('.jsonl.gz')]
                
                for filename in train_files:
                    if max_per_lang > 0 and len(examples) >= max_per_lang:
                        break
                    
                    with zf.open(filename) as zf_file:
                        with gzip.open(zf_file, 'rt', encoding='utf-8') as gz_file:
                            for line in gz_file:
                                if max_per_lang > 0 and len(examples) >= max_per_lang:
                                    break
                                try:
                                    entry = json.loads(line)
                                    # Extract relevant fields
                                    code = entry.get("code", entry.get("func_code_string", ""))
                                    docstring = entry.get("docstring", entry.get("func_documentation_string", ""))
                                    
                                    # Skip entries without docstrings
                                    if code and docstring and len(docstring.strip()) > 10:
                                        examples.append({
                                            "code": code,
                                            "docstring": docstring,
                                            "language": lang,
                                            "repo": entry.get("repo", ""),
                                            "path": entry.get("path", ""),
                                        })
                                except json.JSONDecodeError:
                                    continue
            
            if examples:
                df = pd.DataFrame(examples)
                output_file = csn_dir / f"{lang}.parquet"
                df.to_parquet(output_file)
                
                stats["languages"][lang] = len(examples)
                stats["total_examples"] += len(examples)
                print(f"  ✓ {lang}: {len(examples):,} code-docstring pairs")
            
        except Exception as e:
            print(f"  ✗ {lang}: Failed - {e}")
    
    return stats


def download_code_docstring_corpus(output_dir: Path, max_examples: int = 50000) -> dict:
    """
    Download code_docstring_corpus dataset.
    
    Args:
        output_dir: Directory to save the dataset
        max_examples: Maximum examples to download
    
    Returns:
        Statistics about downloaded data
    """
    from datasets import load_dataset
    
    stats = {"total_examples": 0}
    
    print("\n📦 Downloading Code Docstring Corpus...")
    
    ds_dir = output_dir / "code_docstring_corpus"
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # This dataset uses 'top_level' split (not 'train')
        dataset = load_dataset("teven/code_docstring_corpus", "default", split="top_level")
        
        # Limit examples (0 = unlimited)
        if max_examples > 0 and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))
        
        # Save as parquet
        output_file = ds_dir / "train.parquet"
        dataset.to_parquet(str(output_file))
        
        stats["total_examples"] = len(dataset)
        print(f"  ✓ Downloaded {len(dataset):,} examples")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    return stats


def download_the_stack_smol(output_dir: Path, languages: list = None, max_per_lang: int = 5000) -> dict:
    """
    Download The Stack Smol (small subset of The Stack).
    
    Args:
        output_dir: Directory to save the dataset
        languages: List of languages to download
        max_per_lang: Maximum examples per language
    
    Returns:
        Statistics about downloaded data
    """
    from datasets import load_dataset
    
    languages = languages or ["python"]
    stats = {"total_examples": 0, "languages": {}}
    
    print(f"\n📦 Downloading The Stack Smol for: {languages}")
    
    stack_dir = output_dir / "the_stack_smol"
    stack_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in languages:
        try:
            print(f"  Downloading {lang}...")
            
            # The Stack Smol has language subsets
            dataset = load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True
            )
            
            examples = []
            for example in tqdm(dataset, desc=lang, total=max_per_lang):
                content = example.get("content", "")
                
                # Filter for files with documentation
                has_doc = False
                if lang == "python":
                    has_doc = '"""' in content or "'''" in content
                elif lang in ["javascript", "typescript"]:
                    has_doc = "/**" in content or "///" in content
                elif lang in ["java", "go"]:
                    has_doc = "/**" in content
                else:
                    has_doc = "/**" in content or "///" in content
                
                if has_doc:
                    examples.append({
                        "content": content,
                        "language": lang,
                        "path": example.get("path", ""),
                    })
                
                if max_per_lang > 0 and len(examples) >= max_per_lang:
                    break
            
            if examples:
                df = pd.DataFrame(examples)
                output_file = stack_dir / f"{lang}.parquet"
                df.to_parquet(output_file)
                
                stats["languages"][lang] = len(examples)
                stats["total_examples"] += len(examples)
                print(f"  ✓ {lang}: {len(examples):,} files with documentation")
            else:
                print(f"  ⚠ {lang}: No documented files found")
            
        except Exception as e:
            print(f"  ✗ {lang}: Failed - {e}")
    
    return stats


def download_python_code_instructions(output_dir: Path, max_examples: int = 20000) -> dict:
    """
    Download Python code instructions dataset (instruction-tuning format).
    
    Args:
        output_dir: Directory to save the dataset
        max_examples: Maximum examples
    
    Returns:
        Statistics about downloaded data
    """
    from datasets import load_dataset
    
    stats = {"total_examples": 0}
    
    print("\n📦 Downloading Python Code Instructions...")
    
    ds_dir = output_dir / "python_code_instructions"
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        
        # Limit examples (0 = unlimited)
        if max_examples > 0 and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))
        
        output_file = ds_dir / "train.parquet"
        dataset.to_parquet(str(output_file))
        
        stats["total_examples"] = len(dataset)
        print(f"  ✓ Downloaded {len(dataset):,} instruction-output pairs")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    return stats


def verify_downloads(output_dir: Path) -> bool:
    """Verify that datasets were downloaded correctly."""
    print("\n🔍 Verifying downloads...")
    
    all_valid = True
    total_files = 0
    total_examples = 0
    
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            parquet_files = list(subdir.rglob("*.parquet"))
            if parquet_files:
                count = 0
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf)
                        count += len(df)
                    except:
                        pass
                print(f"  {subdir.name}: {len(parquet_files)} files, {count:,} examples")
                total_files += len(parquet_files)
                total_examples += count
            else:
                print(f"  {subdir.name}: No parquet files found")
                all_valid = False
    
    if total_files == 0:
        print("  ⚠ No datasets downloaded yet")
        all_valid = False
    else:
        print(f"\n  Total: {total_files} files, {total_examples:,} examples")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for AutoDoc fine-tuning"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="codesearchnet,docstring,instructions",
        help="Comma-separated: codesearchnet,docstring,stack-smol,instructions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for downloaded datasets"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="python,javascript,java,go,ruby,php",
        help="Comma-separated list of languages (for codesearchnet)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum examples per dataset (0 = unlimited)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing downloads"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verify:
        verify_downloads(output_dir)
        return
    
    datasets_to_download = [d.strip().lower() for d in args.datasets.split(",")]
    languages = [l.strip() for l in args.languages.split(",")]
    
    print("=" * 50)
    print("AutoDoc Dataset Downloader")
    print("=" * 50)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Datasets: {datasets_to_download}")
    print(f"Max examples: {args.max_examples}")
    
    all_stats = {}
    
    if "codesearchnet" in datasets_to_download:
        all_stats["codesearchnet"] = download_codesearchnet(
            output_dir,
            languages=languages,
            max_per_lang=args.max_examples // len(languages)
        )
    
    if "docstring" in datasets_to_download:
        all_stats["docstring"] = download_code_docstring_corpus(
            output_dir, 
            max_examples=args.max_examples
        )
    
    if "stack-smol" in datasets_to_download:
        all_stats["stack-smol"] = download_the_stack_smol(
            output_dir,
            languages=languages,
            max_per_lang=args.max_examples // len(languages)
        )
    
    if "instructions" in datasets_to_download:
        all_stats["instructions"] = download_python_code_instructions(
            output_dir,
            max_examples=args.max_examples
        )
    
    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    
    total = 0
    for name, stats in all_stats.items():
        examples = stats.get("total_examples", 0)
        print(f"  {name}: {examples:,} examples")
        total += examples
    
    print(f"\n  Total: {total:,} examples")
    
    # Verify
    verify_downloads(output_dir)
    
    print("\n✅ Download complete!")


if __name__ == "__main__":
    main()
