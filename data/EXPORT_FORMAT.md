# Export Format

Output of `pipeline/export.py`. Copied to `site/data/` automatically unless `--no-copy` is passed.

```
export/
  meta.json                     manifest + schema
  geometry.drc                  Draco point cloud, positions only
  attributes.bin.gz             packed per-point attribute arrays
  chunks/
    chunk_000000.json.gz        N_CHUNKS files, up to CHUNK_SIZE recipes each
    chunk_000001.json.gz
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
