# Technical Documentation — Proyecto Similitud

Detailed technical documentation covering the architecture, mathematical foundations, module descriptions, known limitations, and planned improvements.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Module Reference](#module-reference)
5. [Data Flow (Pipeline)](#data-flow-pipeline)
6. [Storage Schema](#storage-schema)
7. [Known Limitations](#known-limitations)
8. [Future Improvements](#future-improvements)

---

## System Overview

Proyecto Similitud is a CLI-based audio similarity search engine. Given a set of input songs (divided into "main" and "optional"), it:

1. Resolves each song name via the iTunes Search API
2. Downloads a 30-second audio preview (AAC format)
3. Converts the preview to WAV (48kHz, mono) using ffmpeg
4. Generates a 512-dimensional audio embedding using CLAP (Contrastive Language-Audio Pretraining)
5. Caches the embedding in a local SQLite database
6. Constructs a weighted query vector from all input embeddings
7. Ranks every song in the catalog by cosine similarity to the query vector
8. Displays the top-N most similar songs

### Why CLAP over traditional audio features?

Traditional approaches (e.g., Librosa-based: tempo, MFCC, chroma) capture low-level acoustic properties but miss perceptual similarity — two songs can have similar tempos and timbres but "feel" completely different.

CLAP embeddings are trained on audio-text pairs, meaning they encode higher-level semantic properties that align more closely with how humans perceive musical similarity (~71-72% agreement with human judgments in benchmarks).

### Why iTunes Search API?

- Public endpoint, no API key or developer account required
- Returns `previewUrl` — a ~30-second AAC audio clip for most songs
- Global catalog coverage (mainstream music)
- Rate limit of ~20 requests/minute (handled via built-in throttling)

---

## Architecture

```
proyecto_similitud/
├── __init__.py                 # Package root (v0.1.0)
├── __main__.py                 # Enables `python -m proyecto_similitud`
├── main.py                     # CLI entry point, imports and exposes `cli`
├── data/
│   ├── itunes_client.py        # iTunes API: search, throttling, retry
│   └── preview_downloader.py   # Download AAC → convert to WAV 48kHz
├── audio/
│   └── embedder.py             # CLAPEmbedder (512-dim) + LibrosaEmbedder (40-dim)
├── similarity/
│   └── engine.py               # Weighted query vector + cosine ranking
├── storage/
│   └── cache.py                # SQLite cache for embeddings + metadata
└── cli/
    └── commands.py             # Click subcommands + Rich display
```

### Dependency Direction

Dependencies flow strictly downward. No circular imports.

```
cli/ ──→ similarity/ ──→ audio/ ──→ data/
  │           │             │
  └───────────┴─────────────┴──→ storage/ (cross-cutting)
```

This design allows a future GUI layer (e.g., Streamlit) to be added on top without modifying the existing pipeline.

---

## Mathematical Foundations

### Audio Embeddings

An audio embedding is a fixed-size numerical vector (in this case, 512 floats) that represents the "essence" of an audio clip in a high-dimensional space. Songs that sound similar to human ears tend to produce vectors that are close together in this space.

The CLAP model (`laion/clap-htsat-unfused`) processes a raw audio waveform and outputs a 512-dimensional vector. Each embedding is L2-normalized to unit length (norm = 1.0).

### Cosine Similarity

Cosine similarity measures the angle between two vectors, ignoring their magnitude:

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Since all embeddings are pre-normalized to unit length (‖A‖ = ‖B‖ = 1.0), cosine similarity simplifies to:

```
cos(A, B) = A · B    (dot product)
```

The result ranges from -1 (opposite) to +1 (identical). In practice, scores for similar songs typically fall in the 0.85–0.95 range.

### Weighted Query Vector

When the user provides multiple input songs, a single query vector is constructed as a weighted linear combination:

```
q = Σ(w_main × e_i) + Σ(w_optional × e_j)
q_normalized = q / ‖q‖
```

Where:
- `e_i` = embedding of each main song (default weight: `w_main = 1.0`)
- `e_j` = embedding of each optional song (default weight: `w_optional = 0.4`)

The final vector is L2-normalized. This means main songs contribute 2.5× more influence than optional songs by default.

**Why this approach?** Main songs define the "core sound" the user wants. Optional songs provide additional context (e.g., mood, genre nuance) without overpowering the query direction.

---

## Module Reference

### `data/itunes_client.py`

| Function | Description |
|:---|:---|
| `search_tracks(query, country, limit)` | Searches iTunes API with throttling (3s between requests) and retry with exponential backoff (3 attempts). |
| `find_best_match(query, country)` | Returns the first search result that has a preview URL. Falls back to the first result if none have previews. |

**Rate limiting:** A global variable tracks the last request timestamp. Each call waits at least 3 seconds since the previous request to stay within iTunes' ~20 req/min limit.

**Retry logic:** On HTTP 429 (rate limit) or connection errors, retries up to 3 times with exponential backoff (2s, 4s, 8s). On HTTP 404, retries are skipped immediately.

### `data/preview_downloader.py`

| Function | Description |
|:---|:---|
| `download_preview(url)` | Downloads an AAC preview and converts to WAV (48kHz, mono) via pydub/ffmpeg. Returns path to temp WAV file. |
| `cleanup_preview(path)` | Deletes the temporary WAV file after embedding extraction. |

### `audio/embedder.py`

| Class | Dimensions | Description |
|:---|:---:|:---|
| `CLAPEmbedder` | 512 | Primary embedder. Loads `laion/clap-htsat-unfused` lazily on first use. CPU-friendly inference. |
| `LibrosaEmbedder` | 40 | Fallback embedder using handcrafted features (MFCC + chroma + spectral contrast + tempo). Lighter but less accurate. |

Both implement the `AudioEmbedder` abstract interface with a single method: `embed(audio_path) → np.ndarray`.

**Lazy loading:** The CLAP model (~600 MB) is loaded only on the first call to `embed()`, not on instantiation. Subsequent calls reuse the loaded model.

### `similarity/engine.py`

| Function | Description |
|:---|:---|
| `build_query_vector(main, optional, w_main, w_optional)` | Builds a normalized query vector from weighted embeddings. |
| `cosine_similarity(a, b)` | Dot product between two unit vectors. |
| `rank_similar(query, catalog, top_n, exclude_ids)` | Ranks all catalog songs by similarity, excluding input songs from results. |

### `storage/cache.py`

| Method | Description |
|:---|:---|
| `get(track_id)` | Retrieves an embedding by iTunes track ID. |
| `put(track_id, ...)` | Stores an embedding + metadata. Uses `INSERT OR REPLACE`. |
| `get_all()` | Returns all embeddings as `{track_id: np.ndarray}`. |
| `get_track_metadata(track_id)` | Returns song metadata without the embedding. |
| `has(track_id)` | Checks if a track is already cached. |
| `count()` | Number of songs in the catalog. |
| `list_tracks()` | Lists all cached songs (metadata only). |

### `cli/commands.py`

Three subcommands built with Click + Rich:

| Command | Description |
|:---|:---|
| `search` | Full pipeline: resolve songs → embed → build query → rank → display results. |
| `catalog` | Batch-add songs to the catalog by artist or search query. |
| `healthcheck` | Validates CLAP functionality: checks ffmpeg, downloads a test preview, runs inference, reports performance metrics. |

---

## Data Flow (Pipeline)

A step-by-step example of what happens when you run:

```bash
python -m proyecto_similitud search -m "Blinding Lights" -m "Save Your Tears" --top 5
```

```
1. CLI parses arguments
   └── main songs: ["Blinding Lights", "Save Your Tears"]
   └── top: 5

2. For each main song:
   a. iTunes API: search("Blinding Lights") → returns track metadata + previewUrl
   b. Check cache: has(track_id)?
      - YES → return cached embedding (skip steps c-e)
      - NO  → continue
   c. Download preview (AAC ~30s) from previewUrl
   d. Convert AAC → WAV (48kHz, mono) via ffmpeg
   e. CLAP inference: WAV → 512-dim embedding (normalized)
   f. Cache: store embedding + metadata in SQLite
   g. Cleanup: delete temporary WAV file

3. Build query vector:
   q = normalize(1.0 × emb_blinding + 1.0 × emb_save)

4. Retrieve all catalog embeddings from SQLite

5. For each catalog song (excluding input songs):
   score = dot(q, catalog_embedding)

6. Sort by score descending, take top 5

7. Display results in Rich table
```

---

## Storage Schema

SQLite database at `data/embeddings.db`:

```sql
CREATE TABLE embeddings (
    track_id      INTEGER PRIMARY KEY,  -- iTunes trackId
    track_name    TEXT NOT NULL,
    artist_name   TEXT NOT NULL,
    album_name    TEXT,
    preview_url   TEXT,
    embedding     BLOB NOT NULL,        -- float32 array as raw bytes
    embedding_dim INTEGER NOT NULL,     -- expected dimension (512 for CLAP)
    model_id      TEXT DEFAULT 'clap-htsat-unfused',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Embeddings are stored as raw byte BLOBs (`np.float32.tobytes()`), reconstructed with `np.frombuffer(..., dtype=np.float32)`. A dimension check on retrieval guards against corruption.

---

## Known Limitations

### Catalog scope
- The catalog only contains songs you explicitly add. There is no pre-built database — you must populate it with the `catalog` command before searching.
- Results are limited to what is in your local catalog. Adding more songs improves result quality.

### Audio source
- Similarity is computed on 30-second previews, not full songs. The preview may not be representative of the entire track (e.g., intros vs. choruses).
- Some songs on iTunes lack a `previewUrl` — these are skipped during catalog building.

### Performance
- First inference is slow (~12-18s) due to model loading. Subsequent inferences are fast (~0.3-0.5s per song).
- The CLAP model uses ~600 MB of RAM once loaded.
- Catalog ranking is brute-force (O(n) dot products). This is fine for hundreds/low thousands of songs but would need a vector index (FAISS) for larger catalogs.

### Search accuracy
- iTunes search may return unexpected matches for ambiguous queries (e.g., "Lucky" matches Britney Spears instead of Radiohead). Adding the artist name helps: `-m "Lucky Radiohead"`.
- The tool does not verify that a returned track is by the intended artist.

### API dependency
- Requires an internet connection for iTunes API calls and first-time model download.
- iTunes rate limits (~20 req/min) slow down large catalog operations. The built-in throttling prevents errors but adds wait time.

---

## Future Improvements

- **GUI layer**: Add a Streamlit or web-based interface on top of the existing pipeline.
- **ONNX Runtime**: Convert CLAP to ONNX format for faster CPU inference.
- **FAISS indexing**: Replace brute-force search with approximate nearest neighbors for larger catalogs.
- **Deezer API fallback**: Use Deezer as a secondary source when iTunes lacks previews.
- **Text-based search**: Leverage CLAP's text encoder to search by description (e.g., "melancholic groove with synths").
- **Batch catalog import**: Support importing from a file (CSV/JSON) with pre-defined song lists.
- **Embedding versioning**: Track which model version generated each embedding for future model upgrades.
