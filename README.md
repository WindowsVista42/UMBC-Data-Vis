# Food.com Recipe Explorer

An interactive 3D visualization of ~230K Food.com recipes where proximity reflects semantic similarity. Recipes are embedded with a sentence transformer, classified into categories via k-NN, and combined with numeric features before being projected into 3D with UMAP. The website takes this data and renders it as a navigable 3D point cloud, with category-based coloring and filtering, on-hover recipe details, and per-recipe breakdowns drawn from the ~1.1M user interactions.

**Live versions:** [Alpha](https://windowsvista42.github.io/recipe_vis) | [Beta](https://windowsvista42.github.io/recipe_vis2/) | [Final](https://windowsvista42.github.io/recipe_explorer)

## Dataset

[Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

## Project structure

```
site/          Three.js web app, loads pre-built data from site/data/
data/          Data processing pipeline
  pipeline/    Per-step scripts (embed, assign, ratings, encode, metrics, project, export)
    configs/   Hand-authored config files (tracked by git)
  artifacts/   All generated intermediate outputs (gitignored)
  export/      Final web app package before copying to site/data/ (gitignored)
  raw/         Downloaded dataset files (gitignored)
_old/          Previous iterations (archived)
```

## Running locally

The site is designed to be hosted as a static site. To run it locally, run the following command in the [`site/`](site/) folder:

```
uv run python -m http.server 8080
```

Open `http://localhost:8080`. `site/data/` must be populated first. Either run the pipeline or copy an existing export into it.

## Reproducing the data / running the pipeline

If you want to regenerate the data yourself, see **[`data/README.md`](data/README.md)**. It covers prerequisites (uv, platform-specific PyTorch), all pipeline steps in order, and how to add new categories or features.

**The short version:** from [`data/`](data/) run `uv run run.py` to run all steps and copy output to `site/data/`.
