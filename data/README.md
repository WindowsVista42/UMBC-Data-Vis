# data

Downloads and prepares the Food.com dataset for the project.

The dataset is sourced from Kaggle: [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

The two files we care about are:

- `RAW_recipes.csv` - recipe metadata. Several fields (`tags`, `nutrition`, `steps`, `ingredients`) are stored as Python list strings in the raw CSV, so `download.py` converts this file to `RAW_recipes.jsonl` with those fields as proper JSON arrays.
- `RAW_interactions.csv` - user ratings and reviews. Flat data, kept as CSV.

## Running

```
uv run download.py
```

This downloads all dataset files into this directory and produces `RAW_recipes.jsonl`.
