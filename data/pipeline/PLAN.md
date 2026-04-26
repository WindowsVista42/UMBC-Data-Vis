# Plan: Recipe Cuisine Classification Pipeline

## Context
We want to take the ~230k recipes in `RAW_recipes.jsonl` and classify each one by cuisine for a 3D point cloud visualization. The approach: embed recipe text using a sentence transformer model, project to 3D with UMAP for display, then separately assign each recipe to the nearest cuisine centroid in the original embedding space.

The pipeline is three independent scripts in `data/pipeline/`. Each script produces files the next one reads. Two config files (cuisine list and field template) live alongside the scripts.

**Scope: only `embed.py` is being implemented now. `project.py` and `assign.py` are documented for future reference.**

## Reference implementations
These existing scripts in `_old/semantic_projection/` should be consulted during implementation:
- `embed_abstracts.py` — reference for the embed step: streaming JSONL, batch encoding with sentence-transformers, GPU detection, saving `.npy` + `_index.json`, tqdm progress
- `project_and_cluster.py` — reference for UMAP projection (two-pass pattern, cosine metric, `low_memory=True`)
- `name_clusters.py` — reference for the assign step structure (loading index, parallel processing pattern)

## Directory structure

```
data/
  pipeline/
    embed.py          # step 1: recipe text -> embeddings  <- implemented
    project.py        # step 2: embeddings -> UMAP 3D coords  (future)
    assign.py         # step 3: embeddings + cuisines -> cuisine labels  (future)
    cuisines.txt      # one cuisine name per line
    embed_config.json # {"template": "{name}. {description}. Ingredients: {ingredients}"}
    PLAN.md           # this file
```

Scripts are run from `data/` (e.g. `uv run pipeline/embed.py`). All outputs land in `data/pipeline/`.

---

## Script 1: `embed.py`

**Input:** `RAW_recipes.jsonl` (default path `../RAW_recipes.jsonl` relative to script)

**Config:** `embed_config.json` — `{"template": "{name}. {description}. Ingredients: {ingredients}"}`. Fields missing from a record are silently skipped (empty string). Easily changed to include/exclude fields.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Loaded via `SentenceTransformer`.

**Process:**
- Stream JSONL line by line
- Apply template to each recipe, skipping missing fields
- Encode in batches (default 64) using `model.encode(..., normalize_embeddings=True)`
- Accumulate into a list, then save as numpy array at the end
- No checkpointing -- just run to completion (~230k recipes, a few minutes on GPU)

**Output:**
- `pipeline/recipes_embeddings.npy` -- shape `(N, 384)` float32, L2-normalized
- `pipeline/recipes_index.json` -- `[{"id": <recipe_id>, "index": <int>}, ...]`

**CLI:**
```
uv run pipeline/embed.py
uv run pipeline/embed.py --max-rows 1000   # for testing
uv run pipeline/embed.py --batch-size 128
```

---

## Script 2: `project.py` (future)

**Input:** `pipeline/recipes_embeddings.npy` + `pipeline/recipes_index.json`

**Process:**
- Single UMAP pass to 3D (visualization only -- cuisine assignment uses raw embeddings, not UMAP coords)
- Parameters: `n_components=3`, `n_neighbors=15`, `min_dist=0.1`, `metric=cosine`

**Output:**
- `pipeline/recipes_umap3d.npy` -- shape `(N, 3)` float32
- `pipeline/recipes_umap3d_index.json` -- `[{"id": <recipe_id>, "index": <int>}, ...]`

**CLI:**
```
uv run pipeline/project.py
uv run pipeline/project.py --max-rows 1000
uv run pipeline/project.py --neighbors 30 --min-dist 0.05
```

---

## Script 3: `assign.py` (future)

**Input:** `pipeline/recipes_embeddings.npy` + `pipeline/recipes_index.json` + `pipeline/cuisines.txt`

**Process:**
- Load cuisine names from `cuisines.txt` (one per line, strip blanks/comments)
- Embed each cuisine name using the same `all-MiniLM-L6-v2` model, L2-normalized
- Cosine similarity = matrix multiply of normalized recipe embeddings x normalized cuisine embeddings -> `(N, M)` matrix
- Per recipe: winner = argmax column; runners-up = next 2 highest scores
- Store winner name + score + top-2 runners-up with their scores

**Output:** `pipeline/recipes_cuisine.json`
```json
[
  {
    "id": 137739,
    "cuisine": "American",
    "score": 0.91,
    "runners_up": [
      {"cuisine": "British", "score": 0.83},
      {"cuisine": "Australian", "score": 0.79}
    ]
  }
]
```

**CLI:**
```
uv run pipeline/assign.py
uv run pipeline/assign.py --cuisines pipeline/cuisines.txt
```

---

## Config files

**`pipeline/cuisines.txt`** -- one cuisine per line, edit freely.

**`pipeline/embed_config.json`** -- template for recipe text. Use `{field_name}` placeholders. List fields (like `ingredients`) are joined with `, `. Missing fields become empty strings.

---

## Dependencies (`data/pyproject.toml`)
- `sentence-transformers` (pulls in torch, transformers, numpy)
- `umap-learn` (for project.py, future)

## Gitignore
`*.npy` is gitignored (large binary outputs).
