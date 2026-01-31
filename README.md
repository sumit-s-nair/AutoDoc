# AutoDoc

> Automatically generate relevant, no-fluff documentation for any codebase using a fine-tuned LLM.

AutoDoc is a **standalone application** that analyzes your codebase and generates high-quality Markdown documentation—READMEs, API docs, and module overviews—without the verbose boilerplate.

## Features

- **Code-Aware Documentation** — Understands code structure, not just syntax
- **No Fluff** — Concise, actionable docs without filler phrases
- **Local-First** — Runs entirely on your machine, no data sent to cloud
- **Language Agnostic** — Works with any programming language
- **Fine-Tuned Model** — Custom-trained Qwen2.5-Coder for documentation tasks

## Project Status

**Phase: Dataset Collection & Fine-Tuning Preparation**

We're currently building and curating a high-quality dataset for fine-tuning. See [docs/DATASET_SOURCES.md](docs/DATASET_SOURCES.md) for details.

### Roadmap

- [x] Research dataset sources
- [x] Design data pipeline
- [ ] Implement dataset collection scripts
- [ ] Data preprocessing & cleaning
- [ ] Fine-tune Qwen2.5-Coder
- [ ] Build standalone Electron app
- [ ] Release v1.0

## Project Structure

```
VS-Code-AutoDoc/
├── docs/                   # Documentation
│   ├── DATASET_SOURCES.md  # Dataset sources & licenses
│   ├── DATA_FORMAT.md      # Training data format
│   └── PIPELINE.md         # Data pipeline documentation
├── scripts/                # Data collection & processing
│   ├── download_datasets.py
│   ├── scrape_github.py
│   └── preprocess.py
├── data/                   # Dataset files (gitignored)
│   ├── raw/
│   └── processed/
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Model | Qwen2.5-Coder (1.5B-7B) |
| Fine-Tuning | Unsloth / Axolotl + LoRA |
| Inference | llama.cpp / Ollama |
| App Framework | Electron |
| Dataset Sources | CodeSearchNet, The Vault, GitHub API |

## Documentation

- [Dataset Sources](docs/DATASET_SOURCES.md) — Where our training data comes from
- [Data Format](docs/DATA_FORMAT.md) — Training data format specification
- [Pipeline](docs/PIPELINE.md) — How the data collection pipeline works

## Contributing

This project is in early development. Contributions welcome once the core pipeline is established.

## License

MIT License - See [LICENSE](LICENSE) for details.
