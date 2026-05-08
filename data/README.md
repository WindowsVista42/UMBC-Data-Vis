# Data Processing

This folder contains the full data pipeline for the Food.com Recipe Explorer. It takes the raw Kaggle dataset and produces the data that the Three.js web app loads from `site/data/`.

The dataset is sourced from Kaggle: [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). It contains around 230k recipes and 1.1 million user interactions collected from Food.com.

The two files we care about are:

- **`raw/RAW_recipes.jsonl`** - one recipe per line. Each record has:
  - `name` - recipe name
  - `description` author provided desription of the recipe
  - `minutes` - number of minutes to complete
  - `contributor_id` - unique author id
  - `submitted` - year/month/day the recipe was posted
  - `ingredients` - list of ingredient strings
  - `steps` - list of instruction strings
  - `tags` - list of tag strings (includes cuisine, meal type, dietary info, etc.)
  - `nutrition` - list of 7 values: calories, total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)
- **`raw/RAW_interactions.csv`** - one row per user interaction. Fields are:
  - `user_id` - unique reviewer id
  - `recipe_id` - unique recipe id
  - `date` - year/month/day the review was left
  - `rating` - integer 1 to 5 (0 means no rating was given, not a zero-star rating)
  - `review` - free text

The pipeline works in stages:

1. **Embedding**: each recipe is encoded as a 384-dimensional semantic vector using a sentence transformer (`all-MiniLM-L6-v2`).
2. **Classification**: recipes are assigned to categories (cuisine, meal type, etc.) using k-NN on the subset of recipes that already have matching Food.com tags. This produces per-class probability vectors used downstream.
3. **Feature encoding**: numeric fields (cook time, rating, review count, etc.) are binned or normalized into feature vectors.
4. **Projection**: embeddings, category probabilities, and encoded features are combined and fed into UMAP to produce a 3D layout where semantically similar recipes cluster together.
5. **Export**: geometry, per-point attributes, and recipe metadata are packaged into compressed files optimized for fast loading and spatial cache locality in the browser, then copied to `site/data/`.

## Prerequisites

**uv** - all scripts are run through `uv`. The installation instructions can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

**GPU / PyTorch** - `pyproject.toml` selects the right wheel automatically:
- **Windows**: CUDA 12.1 wheel from the PyTorch index
- **macOS / Linux**: CPU-only wheel from PyPI

Run `uv sync` once before starting (or let `uv run` do it) and the correct dependencies will be installed for your platform.

## Directory structure

```
data/
  pipeline/           pipeline scripts
    configs/          hand-authored config files (tracked by git)
  artifacts/          all generated outputs (gitignored), removed by clean.py
  export/             final web app package (gitignored), removed by clean.py
  raw/                downloaded dataset files (gitignored)
  pipeline.log        log from the last run.py execution (gitignored)
```

## Running the full pipeline (the easy way)

```
uv run run.py
```

Runs all steps in order. Console output is also written to `pipeline.log` by default. To disable: `uv run run.py --no-log`. To write to a different path: `uv run run.py --log path/to/file.log`.

To start from a specific step (skipping earlier ones):

```
uv run run.py --from embed
uv run run.py --from project
```

Available steps: `download`, `embed`, `assign`, `ratings`, `encode`, `metrics`, `project`, `export`

`run.py` reads `config.json` for all pipeline settings and checks that every referenced config file exists before starting. The export step auto-copies output to `../site/data/`.

To remove all generated outputs:

```
uv run clean.py           # removes artifacts/ and export/
uv run clean.py --dry-run # preview what would be removed
```

## Running the full pipeline (the hard way)

If you want to run every step manually...

### 1. Download

```
uv run download.py
```

Downloads all dataset files into `raw/` and produces `raw/RAW_recipes.jsonl`.

### 2. Embed

```
uv run pipeline/embed.py
```

Encodes each recipe as a 384-dim vector using `all-MiniLM-L6-v2`. The text fed to the model is controlled by `pipeline/configs/embed_config.json`. Uses GPU automatically if available. Outputs to `artifacts/`:

- `recipes_embeddings.npy` - shape `(N, 384)`, L2-normalized
- `recipes_index.json` - maps row index to recipe ID

### 3. Assign categories

```
uv run pipeline/derive_taxonomy.py pipeline/configs/cuisines.txt
uv run pipeline/assign.py pipeline/configs/cuisines.txt

uv run pipeline/derive_taxonomy.py pipeline/configs/meal_types.txt
uv run pipeline/assign.py pipeline/configs/meal_types.txt
```

Uses k-NN classification (k=100, distance-weighted, cosine) seeded by Food.com's own tags to classify all recipes. `derive_taxonomy.py` derives parent-child tag relationships from co-occurrence data so the most specific label is used for training. A recipe tagged both `american` and `southern-united-states` trains as `southern-united-states`. Prior correction removes class imbalance bias from the k-NN posterior.

`derive_taxonomy.py` must be run before `assign.py` for each category file. Without the taxonomy, recipes with multiple matching tags are treated as ambiguous and excluded from training.

Each run outputs to `artifacts/`:
- `recipes_{name}.json` - per-recipe top assignment with score and runners-up
- `recipes_{name}_proba.npy` - full probability matrix for UMAP augmentation
- `recipes_{name}_classes.json` - ordered list of category labels

To find what tags are available for a given parent:
```
uv run pipeline/cuisine_tags.py                    # cuisine tags
uv run pipeline/cuisine_tags.py --parent course    # meal/course tags
```

### 4. Process ratings

```
uv run pipeline/process_ratings.py
```

Computes per-recipe `avg_rating` and `n_ratings` from `raw/RAW_interactions.csv`, excluding zero ratings. Writes `artifacts/recipe_contrib_ratings.json.gz` which `export.py` picks up automatically.

### 5. Encode additional features

Each script produces a `*_features.npy` file in `artifacts/` that `project.py` picks up automatically.

**Numeric (soft bins or normalized):**
```
uv run pipeline/encode_ordinal.py pipeline/configs/minutes.json
uv run pipeline/encode_ordinal.py pipeline/configs/n_steps.json
uv run pipeline/encode_ordinal.py pipeline/configs/n_ingredients.json
uv run pipeline/encode_ordinal.py pipeline/configs/submitted.json
```

**From a contrib file (e.g. ratings not in the JSONL):**
```
uv run pipeline/encode_ordinal.py pipeline/configs/avg_rating.json --contrib artifacts/recipe_contrib_ratings.json.gz
uv run pipeline/encode_ordinal.py pipeline/configs/n_ratings.json  --contrib artifacts/recipe_contrib_ratings.json.gz
```

Encoding types: `bins` (hard boundaries with rank-hot output by default), `normalize` (min-max scalar).

### 6. Metrics

```
uv run pipeline/metrics.py
```

Generates per-recipe and per-category metric files from `raw/RAW_interactions.csv`. Outputs to `artifacts/`:

- `recipe_metrics/{shard}.json.gz` - 100 sharded files (keyed by last 2 digits of recipe ID). Each file is a JSON object mapping recipe ID string to `{avg_rating, n_reviews, n_ratings, count_5..count_1, n_per_year}`. Recipes with no interactions are stored as `{}`.
- `category_metrics/{family}_{label}.json.gz` - one file per category, containing aggregate rating stats and top ingredients
- `category_metrics/index.json` - maps each `(family, label)` pair to its filename for client-side lookup

### 7. Project

```
uv run pipeline/project.py
```

Runs UMAP on the embeddings augmented with all category and feature vectors found in `artifacts/`. Weights for each input are tunable in `pipeline/configs/projection_weights.json`. Outputs to `artifacts/`:
- `recipes_umap3d.npy` and `recipes_umap3d_index.json`

Pass `--dims 2` for a 2D projection.

### 8. Export

```
uv run pipeline/export.py
```

Packages everything into the format the web app expects. Outputs to `export/` and automatically copies to `../site/data/`:
- `geometry.drc` - Draco-compressed 3D positions
- `attributes.bin.gz` - per-point attributes: recipe ID, chunk ID, category IDs, scalar features
- `meta.json` - manifest: totals, category families with labels, attribute layout
- `chunks/chunk_XXXXXX.json.gz` - recipe metadata, spatially organized by Z-order curve for good hover cache locality
- `recipe_metrics/` and `category_metrics/` - copied from `artifacts/` (requires the metrics step to have been run first)

Points are sorted by Z-order (Morton code) so nearby points in 3D space land in the same metadata chunk.

To add extra metadata fields to the chunks: write a `artifacts/recipe_contrib_*.json.gz` (keyed by recipe ID string) and re-run `export.py`. No other changes needed.

See `EXPORT_FORMAT.md` for a detailed description of the output format.

## Config files

- **`config.json`** - top-level pipeline config read by `run.py`:
  - `embed_config` - path to the embed config
  - `projection_weights` - path to the projection weights
  - `assign` - list of `.txt` files to run assign on
  - `encode` - list of `{config, contrib?}` objects for numeric encoding
- **`pipeline/configs/embed_config.json`** - template for recipe text passed to the embedding model:
  - `template` - replaces the `{field}` placeholders with the text of that field for each recipe.
- **`pipeline/configs/cuisines.txt`** - cuisine tag names for `assign.py`:
  - One tag per line, must match Food.com tag names exactly (lowercase, hyphenated)
- **`pipeline/configs/meal_types.txt`** - meal type tag names for `assign.py`:
  - Same format as `cuisines.txt`
- **`pipeline/configs/projection_weights.json`** - multiplicative weights for each input to the UMAP projection:
  - Keys are file stems (e.g. `"cuisines_proba"`, `"n_steps_features"`)
  - Use `"embeddings"` to scale the base embedding vectors
  - Use `"default"` as a fallback for any file not explicitly listed
- **`pipeline/configs/minutes.json`, `n_steps.json`, `n_ingredients.json`, `submitted.json`, `avg_rating.json`, `n_ratings.json`** - numeric encoding configs:
  - Each specifies a `field`, encoding `type` (`bins` or `normalize`), and bin edges or label

## Adding a new category

1. Run `pipeline/cuisine_tags.py --parent <parent-tag>` to find valid tag names
2. Create a new `.txt` file in `pipeline/configs/` with the tags you want
3. Add it to the `assign` list in `config.json`
4. Run `uv run run.py --from assign`

## Adding a new numeric feature

1. Create a JSON config in `pipeline/configs/` with the field name and bin edges
2. Add it to the `encode` list in `config.json` (include `contrib` if the field comes from a contrib file)
3. Add a weight for it in `pipeline/configs/projection_weights.json`
4. Run `uv run run.py --from encode`
