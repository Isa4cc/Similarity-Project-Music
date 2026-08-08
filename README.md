# 🎵 Proyecto Similitud — Audio Similarity Search Engine

A personal CLI application that finds similar songs using CLAP audio embeddings and cosine similarity ranking.

Instead of relying on streaming platform recommendations, this tool gives you full control over how "similar" is defined — by analyzing the actual audio signal of songs you choose.

## Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and available in PATH (required for audio conversion)

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Verify installation
ffmpeg -version
```

## Installation

```bash
# Clone and install in development mode
cd Project-music-line
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> **Note:** The first time CLAP is used, the model will be downloaded automatically (~600 MB). This only happens once and is cached by Hugging Face.

## Quick Start

### 1. Verify CLAP works on your hardware

```bash
python -m proyecto_similitud healthcheck
```

This downloads a test preview, runs CLAP inference, and reports timing + RAM usage.

### 2. Build your song catalog

Before searching for similar songs, you need to populate your catalog:

```bash
# Add songs by artist
python -m proyecto_similitud catalog --artist "Radiohead" --limit 20

# Add songs from another artist
python -m proyecto_similitud catalog --artist "The Weeknd" --limit 20

# Add songs by generic search
python -m proyecto_similitud catalog --query "alternative rock 90s" --limit 15
```

### 3. Search for similar songs

```bash
python -m proyecto_similitud search \
  -m "Karma Police" \
  -m "Paranoid Android" \
  -m "Lucky" \
  -m "Let Down" \
  -m "No Surprises" \
  -o "Everything In Its Right Place" \
  -o "Idioteque" \
  --top 10
```

> **Important:** Each song requires its own `-m` (main) or `-o` (optional) flag. You cannot pass multiple song names after a single flag.

### Advanced Options

```bash
# Custom weighting
python -m proyecto_similitud search \
  -m "Song 1" \
  -m "Song 2" \
  -m "Song 3" \
  --weight-main 1.0 \
  --weight-optional 0.3 \
  --top 15

# Specify iTunes country store (default: US)
python -m proyecto_similitud search -m "Song" --country MX
```

## CLI Reference

### Global Options

| Option | Description |
|:---|:---|
| `--verbose` / `-v` | Show detailed logs |
| `--db PATH` | Path to embeddings database (default: `data/embeddings.db`) |

### `search` — Find similar songs

| Option | Alias | Required | Description |
|:---|:---:|:---:|:---|
| `--main` | `-m` | ✅ | Main input songs (repeat flag per song, ~5 recommended) |
| `--optional` | `-o` | No | Optional context songs (repeat flag per song) |
| `--top` | `-n` | No | Number of results (default: 10) |
| `--country` | — | No | iTunes country code (default: US) |
| `--weight-main` | — | No | Weight for main songs (default: 1.0) |
| `--weight-optional` | — | No | Weight for optional songs (default: 0.4) |

### `catalog` — Populate the song catalog

| Option | Alias | Required | Description |
|:---|:---:|:---:|:---|
| `--artist` | `-a` | One of `--artist` or `--query` | Search songs by artist name |
| `--query` | `-q` | One of `--artist` or `--query` | Generic iTunes search |
| `--limit` | `-l` | No | Number of songs to add (default: 10) |
| `--country` | — | No | iTunes country code (default: US) |

### `healthcheck` — Verify CLAP on your hardware

No additional options. Reports: embedding dimension, L2 norm, inference time, and RAM usage.

## Architecture

```
proyecto_similitud/
├── main.py              # CLI entry point
├── data/                # iTunes API client + preview downloader
├── audio/               # CLAP model + embedding generation
├── similarity/          # Cosine similarity + weighted ranking
├── storage/             # SQLite embedding cache
└── cli/                 # Click commands + Rich output formatting
```

**Pipeline:** `CLI → data/ (fetch) → audio/ (embed) → storage/ (cache) → similarity/ (rank) → CLI (display)`

For detailed technical documentation, see [DOCS.md](DOCS.md).

## Development

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_cache.py -v
```

## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.
