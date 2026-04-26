# data

Downloads the Food.com dataset and runs a pipeline to embed recipes and project them into 3D space for visualization.

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
- `rating` - integer 1 to 5
- `review` - free text

---

## Pipeline

The pipeline runs in this order. Steps 3-5 can be re-run independently and in any combination once the embeddings exist.

### 1. Download

```
uv run download.py
```

Downloads all dataset files into `raw/` and produces `raw/RAW_recipes.jsonl`.

### 2. Embed

```
uv run pipeline/embed.py
```

Encodes each recipe into a 384-dim vector using `all-MiniLM-L6-v2`. The text fed to the model is controlled by `pipeline/embed_config.json`. Uses GPU automatically if available. Outputs:

- `pipeline/recipes_embeddings.npy` - shape `(N, 384)`, L2-normalized
- `pipeline/recipes_index.json` - maps row index to recipe ID

### 3. Assign categories

```
uv run pipeline/assign.py pipeline/cuisines.txt
uv run pipeline/assign.py pipeline/meal_types.txt
```

Uses k-nearest-neighbor classification (k=100, distance-weighted) to assign each recipe to a category. Training data is recipes that are already tagged with one of the categories in the `.txt` file. The taxonomy file (derived in step 3a) ensures that when a recipe has multiple matching tags, the most specific one is used.

Applies prior correction to remove class imbalance bias from the KNN posterior. Tagged recipes get leave-one-out probabilities so their own label does not dominate their neighborhood.

Each run outputs:
- `pipeline/recipes_{name}.json` - per-recipe assignment with score and two runners-up
- `pipeline/recipes_{name}_proba.npy` - full `(N, n_classes)` probability matrix used by project.py
- `pipeline/recipes_{name}_classes.json` - ordered class list matching proba columns

#### 3a. Derive taxonomy (run once per category file)

```
uv run pipeline/derive_taxonomy.py pipeline/cuisines.txt
uv run pipeline/derive_taxonomy.py pipeline/meal_types.txt
```

Scans the JSONL to find parent-child relationships between tags (e.g. `cajun` is always a subset of `southern-united-states`). Outputs `pipeline/{name}_taxonomy.json` which assign.py reads automatically. Also outputs a human-readable `pipeline/{name}_taxonomy.txt` to cross-reference.

#### 3b. List available tags (optional helper)

```
uv run pipeline/cuisine_tags.py                    # cuisine tags
uv run pipeline/cuisine_tags.py --parent course    # meal/course tags
uv run pipeline/cuisine_tags.py --parent main-ingredient
```

Lists all tags that exclusively appear under a given parent tag, with recipe counts. Use this to decide what goes in a `.txt` category file.

### 4. Encode additional features

These produce feature vectors that project.py picks up automatically.

#### Tag-based (hard one-hot)

```
uv run pipeline/tag_encode.py pipeline/cook_time.txt --other longer
```

Assigns each recipe to the first matching tag in the list (most specific first). Recipes with no match get the `--other` label. Outputs `pipeline/recipes_{name}_proba.npy`.

#### Numeric (soft-binned or normalized)

```
uv run pipeline/numeric_encode.py pipeline/minutes.json
uv run pipeline/numeric_encode.py pipeline/n_steps.json
uv run pipeline/numeric_encode.py pipeline/submitted.json
```

Encodes a single numeric field per config file. Two types:

- `bins` - soft binning with linear interpolation between bin centers. A recipe halfway between two centers gets split weight. Good for step counts, cook time.
- `normalize` - min-max normalization to [0, 1] as a single column. Good for dates and other continuous values where bin placement would be arbitrary.

Outputs `pipeline/recipes_{name}_features.npy`.

### 5. Project

```
uv run pipeline/project.py
```

Runs UMAP on the embeddings to produce 3D coordinates for visualization. Auto-detects all `*_proba.npy` and `*_features.npy` files in the pipeline directory and concatenates them to the embeddings before projection. This pulls recipes with similar category assignments together in the 3D space.

Key flags:
- `--dims 2` - produce a 2D projection instead
- `--category-weight 0.5` - how strongly category features influence the projection (default: 0.5, set to 0 to disable)
- `--neighbors 15` - UMAP n_neighbors
- `--min-dist 0.1` - UMAP min_dist
- `--random-state 42` - fix seed for reproducibility

Outputs:
- `pipeline/recipes_umap3d.npy` - shape `(N, 3)`
- `pipeline/recipes_umap3d_index.json`

### 6. Preview

```
uv run preview.py
```

Generates `preview.html`, a standalone interactive plotly point cloud. Open it in any browser. Colors by cuisine if assign.py has been run, otherwise uniform. Hover over a point to see recipe name, cuisine, cook time, ingredients, and description.

Key flags:
- `--max-rows 10000` - sample a subset for faster rendering
- `--dims 2` - force 2D (auto-detects 3D by default)
- `--assignment pipeline/recipes_cuisines.json` - which assignment file to use for coloring
- `--seed 42` - reproducible point sampling

---

## Config files

**`pipeline/embed_config.json`** - template for the text passed to the embedding model. Uses `{field}` placeholders. List fields like `ingredients` are joined with `, `. Edit to include or exclude recipe fields.

**`pipeline/cuisines.txt`** - cuisine categories for assign.py. One tag name per line, must match Food.com tag names exactly (lowercase, hyphenated). Run `cuisine_tags.py` to find valid names.

**`pipeline/meal_types.txt`** - meal type categories for assign.py. Same format.

**`pipeline/minutes.json`** - soft bin config for cook time in minutes.

**`pipeline/n_steps.json`** - soft bin config for number of steps.

**`pipeline/submitted.json`** - normalize config for submission date.

---

## Adding a new category

1. Run `cuisine_tags.py --parent <parent-tag>` to find valid tag names under that parent
2. Create a new `.txt` file with the tags you want
3. Run `derive_taxonomy.py` on it
4. Run `assign.py` on it
5. Rerun `project.py` to incorporate the new proba file
