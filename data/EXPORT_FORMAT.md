# Export Format

Output of `pipeline/export.py`. Copied to `site/data/` automatically unless `--no-copy` is passed. `recipe_metrics/` and `category_metrics/` are copied from `artifacts/` if present (requires the `metrics` pipeline step to have been run first).

```
export/
  meta.json                           manifest + schema
  geometry.drc                        Draco point cloud, positions only
  attributes.bin.gz                   packed per-point attribute arrays
  chunks/
    chunk_000000.json.gz              N_CHUNKS files, up to CHUNK_SIZE recipes each
    chunk_000001.json.gz
    ...
  recipe_metrics/
    00.json.gz                        100 shard files (00..99) keyed by last 2 digits of recipe ID
    01.json.gz
    ...
  category_metrics/
    index.json                        maps (family, label) to filename
    avg_rating_no-rating.json.gz      one file per category
    cuisines_american.json.gz
    ...
```

All files share a single **Z-order (Morton code) sort**: index `i` refers to the same recipe in `geometry.drc`, `attributes.bin.gz`, and the chunk assignment.

---

## meta.json

Read this first. Describes all other files.

```json
{
  "total": 231637,
  "n_chunks": 464,
  "chunk_size": 500,
  "coord_bounds": {
    "min": [x_min, y_min, z_min],
    "max": [x_max, y_max, z_max]
  },
  "attribute_layout": [
    { "name": "recipe_id", "dtype": "uint32" },
    { "name": "chunk_id",  "dtype": "uint16" },
    { "name": "avg_rating","dtype": "uint8"  },
    ...
  ],
  "categories": [
    {
      "name": "avg_rating",
      "attribute": "avg_rating",
      "labels": ["no rating", "1 star", "2 stars", "3 stars", "4 stars", "4.5 stars", "4.8 stars", "5 stars"]
    },
    ...
  ],
  "scalar_attributes": []
}
```

`coord_bounds` gives the UMAP bounding box, used to position the camera on load.

`attribute_layout` is the authoritative description of `attributes.bin.gz`. Each entry is one contiguous block of `total` values.

`categories`: each entry maps a `name` (used in the UI) to an `attribute` (the field name in `attribute_layout`) and an ordered `labels` array. The uint8 value stored per point is the index into `labels`.

---

## geometry.drc

Draco-compressed positions for all N points, in Z-order. Decoded with Three.js `DRACOLoader`.

- 16-bit quantized XYZ floats
- `preserve_order: true`, so decoded index matches Z-order

Positions only. Per-point attributes are in `attributes.bin.gz` to allow better compression with native integer types.

---

## attributes.bin.gz

Gzip-compressed binary blob. Contains one contiguous block per entry in `meta.json -> attribute_layout`, each block being exactly `N * sizeof(dtype)` bytes in Z-order. No headers or padding between blocks.

| dtype | bytes/point | notes |
|-------|-------------|-------|
| `uint32` | 4 | `recipe_id`: Food.com recipe ID |
| `uint16` | 2 | `chunk_id`: index of the chunk file holding this recipe's metadata |
| `uint8` | 1 | category label index, one block per category family |
| `float32` | 4 | scalar attribute, one block per scalar (none currently) |

Reading all blocks in Python:

```python
import numpy as np, gzip, json

meta = json.load(open("meta.json"))
N    = meta["total"]
sizes  = {"uint32": 4, "uint16": 2, "uint8": 1, "float32": 4}
dtypes = {"uint32": np.uint32, "uint16": np.uint16, "uint8": np.uint8, "float32": np.float32}

raw = gzip.open("attributes.bin.gz", "rb").read()
offset, attrs = 0, {}
for attr in meta["attribute_layout"]:
    nb = N * sizes[attr["dtype"]]
    attrs[attr["name"]] = np.frombuffer(raw[offset:offset+nb], dtype=dtypes[attr["dtype"]])
    offset += nb
```

---

## chunks/chunk_XXXXXX.json.gz

Gzip-compressed JSON files, each holding up to `chunk_size` recipes. Keys are recipe ID strings.

```json
{
  "134728": {
    "name": "Copycat KFC Chicken",
    "description": "A copycat recipe...",
    "ingredients": ["flour", "paprika", "garlic salt", ...],
    "minutes": 90,
    "n_steps": 8,
    "n_ingredients": 11,
    "submitted": "2005-04-12",
    "avg_rating": 4.833,
    "n_ratings": 6
  },
  ...
}
```

| Field | Source | Notes |
|-------|--------|-------|
| `name` | `RAW_recipes.jsonl` | |
| `description` | `RAW_recipes.jsonl` | May be empty string |
| `ingredients` | `RAW_recipes.jsonl` | List of strings |
| `minutes` | `RAW_recipes.jsonl` | |
| `n_steps` | `RAW_recipes.jsonl` | |
| `n_ingredients` | `RAW_recipes.jsonl` | |
| `submitted` | `RAW_recipes.jsonl` | `YYYY-MM-DD` |
| `avg_rating` | `recipe_contrib_ratings.json.gz` | Absent if recipe has no ratings |
| `n_ratings` | `recipe_contrib_ratings.json.gz` | Absent if recipe has no ratings |

Chunk contents are derived from `RAW_recipes.jsonl` (base fields) merged with any `artifacts/recipe_contrib_*.json.gz` files, so adding a new contrib file and re-running export will add its fields to the chunks automatically.

Points are Z-order sorted before chunking, so spatially nearby UMAP points land in the same chunk. In practice, hovering over a region of the visualization loads one or two chunks.

Use the `chunk_id` from `attributes.bin.gz` to find a recipe's chunk. Do not recompute it from the point index.

---

## recipe_metrics/{shard}.json.gz

100 gzip-compressed JSON files, sharded by the last 2 digits of the recipe ID (`00` through `99`). Each file is a JSON object keyed by recipe ID string.

```json
{
  "228132": {
    "avg_rating": 5.0,
    "n_reviews": 14,
    "n_ratings": 13,
    "count_5": 13,
    "count_4": 0,
    "count_3": 0,
    "count_2": 0,
    "count_1": 0,
    "n_per_year": { "2007": 3, "2008": 5 }
  },
  "228134": {},
  ...
}
```

| Field | Notes |
|-------|-------|
| `avg_rating` | Mean of non-zero ratings |
| `n_reviews` | Total interactions including 0-rated |
| `n_ratings` | Interactions with rating > 0 |
| `count_1..count_5` | Per-star rating counts |
| `n_per_year` | All interactions by year (including 0-rated) |

Recipes with no interactions are stored as `{}`. Shard for a recipe: last 2 characters of the ID string (e.g. recipe `228132` is in `32.json.gz`).

---

## category_metrics/index.json

Plain JSON (not compressed). Maps each `(family, label)` pair to its filename for direct client-side lookup without string manipulation.

```json
{
  "avg_rating": {
    "no rating":  "avg_rating_no-rating.json.gz",
    "1 star":     "avg_rating_1-star.json.gz",
    "4.5 stars":  "avg_rating_4.5-stars.json.gz",
    ...
  },
  "cuisines": {
    "american":   "cuisines_american.json.gz",
    ...
  }
}
```

---

## category_metrics/{family}_{slug}.json.gz

One gzip-compressed JSON file per category. The filename is derived by slugifying the label (`" "` to `"-"`, `"+"` to `"plus"`, `"<"` to `"lt"`).

```json
{
  "family": "cuisines",
  "category": "american",
  "recipe_count": 30412,
  "reviews_per_year": { "2003": 812, "2004": 2341 },
  "avg_rating": 4.21,
  "total_reviews": 94832,
  "ingredients": { "butter": 8423, "salt": 7901, "garlic": 6244 }
}
```

| Field | Notes |
|-------|-------|
| `recipe_count` | Recipes assigned to this category |
| `reviews_per_year` | All interactions by year (including 0-rated) |
| `avg_rating` | Mean of non-zero ratings across the category |
| `total_reviews` | Total interactions including 0-rated |
| `ingredients` | Ingredient occurrence counts across all recipes in the category, sorted descending |
