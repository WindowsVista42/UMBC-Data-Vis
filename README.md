# Food.com Recipe Explorer

Interactive 3D visualization of 230k recipes from Food.com, projected into embedding space using sentence transformers and UMAP.

Live versions: [Alpha](https://windowsvista42.github.io/recipe_vis) | [Beta](https://windowsvista42.github.io/recipe_vis2/)

## Project structure

```
site/        Three.js web app, loads pre-built data from site/data/
data/        Pipeline: downloads dataset, builds embeddings, exports to site/data/
  pipeline/  Per-step scripts (embed, assign, project, export, etc.)
  raw/       Downloaded dataset files (gitignored)
  export/    Pipeline output before copying to site/data/ (gitignored)
_old/        Previous iterations (archived)
```

## Running locally

The site is designed to be hosted as a static site. To run it locally, run the following command in the [`site/`](site/) folder:

```
uv run python -m http.server 8080
```

Open `http://localhost:8080`. `site/data/` must be populated first. Either run the pipeline or copy an existing export into it.

## Data pipeline

See [`data/README.md`](data/README.md) for full documentation: pipeline steps, config files, and how to add new categories or features.

From [`data/`](data/), `uv run run.py` runs all steps end-to-end and copies output to `site/data/`.

## Dataset

The pipeline downloads [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) from Kaggle automatically via the `download` step.
