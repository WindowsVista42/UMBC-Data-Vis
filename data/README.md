# data

Downloads the Food.com dataset and runs a pipeline to embed recipes, classify them by various dimensions, and project them into 3D space for visualization.

The dataset is sourced from Kaggle: [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). It contains around 230k recipes and 1.1 million user interactions collected from Food.com.

The two files we care about are:

**`raw/RAW_recipes.jsonl`** - one recipe per line. Each record has:
- `name`, `description`, `minutes`, `contributor_id`, `submitted`
- `ingredients` - list of ingredient strings
- `steps` - list of instruction strings
- `tags` - list of tag strings (includes cuisine, meal type, dietary info, etc.)
- `nutrition` - list of 7 values: calories, total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)

Converted from CSV on download so that all list fields are proper JSON arrays rather than Python string literals.

**`raw/RAW_interactions.csv`** - one row per user interaction. Fields are:
- `user_id`, `recipe_id`, `date`
- `rating` - integer 1 to 5 (0 means no rating was given, not a zero-star rating)
- `review` - free text

---

## Pipeline

Run steps in order. Steps 3-7 can be re-run independently once the embeddings exist.

### 1. Download

```
uv run download.py
```

Downloads all dataset files into `raw/` and produces `raw/RAW_recipes.jsonl`.

### 2. Embed

```
uv run pipeline/embed.py
```

Encodes each recipe as a 384-dim vector using `all-MiniLM-L6-v2`. The text fed to the model is controlled by `pipeline/embed_config.json`. Uses GPU automatically if available. Outputs:

- `pipeline/recipes_embeddings.npy` - shape `(N, 384)`, L2-normalized
- `pipeline/recipes_index.json` - maps row index to recipe ID

### 3. Assign categories

```
uv run pipeline/derive_taxonomy.py pipeline/cuisines.txt
uv run pipeline/assign.py pipeline/cuisines.txt

uv run pipeline/derive_taxonomy.py pipeline/meal_types.txt
uv run pipeline/assign.py pipeline/meal_types.txt
```

Uses k-NN classification (k=100, distance-weighted, cosine) seeded by Food.com's own tags to classify all recipes. `derive_taxonomy.py` derives parent-child tag relationships from co-occurrence data so the most specific label is used for training. Prior correction removes class imbalance bias from the k-NN posterior.

Each run outputs `pipeline/recipes_{name}.json` (per-recipe assignment), `pipeline/recipes_{name}_proba.npy` (full probability matrix for UMAP), and `pipeline/recipes_{name}_classes.json`.

To find what tags are available for a given parent:
```
uv run pipeline/cuisine_tags.py                    # cuisine tags
uv run pipeline/cuisine_tags.py --parent course    # meal/course tags
```

### 4. Process ratings

```
uv run pipeline/process_ratings.py
```

Computes per-recipe `avg_rating` and `n_ratings` from `raw/RAW_interactions.csv`, excluding zero ratings. Writes `pipeline/recipe_contrib_ratings.json.gz` which `export.py` picks up automatically.

### 5. Encode additional features

Each script produces a `*_features.npy` file that `project.py` picks up automatically.

**Tag-based:**
```
uv run pipeline/tag_encode.py pipeline/cook_time.txt --other longer
```

**Numeric (soft bins or normalized):**
```
uv run pipeline/numeric_encode.py pipeline/minutes.json
uv run pipeline/numeric_encode.py pipeline/n_steps.json
uv run pipeline/numeric_encode.py pipeline/n_ingredients.json
uv run pipeline/numeric_encode.py pipeline/submitted.json
```

**From a contrib file (e.g. ratings not in the JSONL):**
```
uv run pipeline/numeric_encode.py pipeline/avg_rating.json --contrib pipeline/recipe_contrib_ratings.json.gz
uv run pipeline/numeric_encode.py pipeline/n_ratings.json  --contrib pipeline/recipe_contrib_ratings.json.gz
```

See `pipeline/PLAN.md` for the encoding types (`bins` vs `normalize`) and how to write new config files.

### 6. Project

```
uv run pipeline/project.py
```

Runs UMAP on the embeddings augmented with all category and feature vectors. Weights for each input are tunable in `pipeline/projection_weights.json`. Outputs `pipeline/recipes_umap3d.npy` and `pipeline/recipes_umap3d_index.json`. Pass `--dims 2` for a 2D projection.

### 7. Export

```
uv run pipeline/export.py
```

Packages everything into the format the web app expects. Outputs to `export/`:
- `geometry.drc` - Draco-compressed 3D positions (0.88 MB)
- `attributes.bin.gz` - per-point attributes: recipe ID, chunk ID, category IDs, scalar features (1.05 MB)
- `meta.json` - manifest: totals, category families with labels, attribute layout
- `chunks/chunk_XXXXXX.json.gz` - recipe metadata, spatially organized by Z-order curve for good hover cache locality (~32 MB total)

Points are sorted by Z-order (Morton code) so nearby points in 3D space land in the same metadata chunk.

To add extra metadata fields: write a `pipeline/recipe_contrib_*.json.gz` (keyed by recipe ID string) and re-run `export.py`. No other changes needed.

### 8. Copy to site

```
cp -r export/* ../site/data/
```

---

## Config files

**`pipeline/embed_config.json`** - template for recipe text passed to the embedding model. Uses `{field}` placeholders; list fields are joined with `, `.

**`pipeline/cuisines.txt`** - cuisine tag names for `assign.py`, one per line. Must match Food.com tag names exactly (lowercase, hyphenated).

**`pipeline/meal_types.txt`** - meal type tag names for `assign.py`.

**`pipeline/projection_weights.json`** - multiplicative weights for each input to the UMAP projection. Keys are file stems (e.g. `"cuisines_proba"`, `"n_steps_features"`). Use `"embeddings"` to scale the base embedding vectors. Use `"default"` as a fallback for any file not explicitly listed.

**`pipeline/minutes.json`, `pipeline/n_steps.json`, `pipeline/n_ingredients.json`, `pipeline/submitted.json`, `pipeline/avg_rating.json`, `pipeline/n_ratings.json`** - numeric encoding configs. Each specifies a field, encoding type (`bins` or `normalize`), and bin centers or label.

---

## Adding a new category

1. Run `cuisine_tags.py --parent <parent-tag>` to find valid tag names
2. Create a new `.txt` file with the tags you want
3. Run `derive_taxonomy.py` on it
4. Run `assign.py` on it
5. Rerun `project.py` and `export.py`

## Adding a new numeric feature

1. Create a JSON config in `pipeline/` with the field name and bin centers
2. Run `numeric_encode.py` on it (add `--contrib` if the field comes from a contrib file)
3. Add a weight for it in `projection_weights.json`
4. Rerun `project.py` and `export.py`
