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

## Steps

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

### 3. Project

```
uv run pipeline/project.py
```

Runs UMAP on the embeddings to produce 3D coordinates for visualization. Takes 10-30 minutes on CPU. Pass `--dims 2` for a 2D projection. Outputs:

- `pipeline/recipes_umap3d.npy` - shape `(N, 3)`
- `pipeline/recipes_umap3d_index.json`

### 4. Assign cuisines

```
uv run pipeline/assign.py
```

Embeds each cuisine name from `pipeline/cuisines.txt` and assigns each recipe to the nearest one by cosine similarity. Outputs `pipeline/recipes_cuisine.json` with the winning cuisine, its similarity score, and the two next closest.

### 5. Preview

```
uv run preview.py
```

Generates `preview.html`, a standalone interactive plotly point cloud. Open it in any browser. Colors by cuisine if step 4 has been run, otherwise uniform. Hover over a point to see recipe details. Defaults to 20k sampled points, change with `--sample`.

## Config

**`pipeline/embed_config.json`** - template for the text passed to the embedding model. Uses `{field}` placeholders. List fields like `ingredients` are joined with `, `. Edit this to include or exclude recipe fields.

**`pipeline/cuisines.txt`** - one cuisine name per line. Edit before running step 4 to change which cuisines are used.
