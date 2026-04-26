# Design Decisions

High-level decisions made during development and the reasoning behind them.

---

## Embedding model: all-MiniLM-L6-v2

We use `sentence-transformers/all-MiniLM-L6-v2` (384-dim) rather than the SPECTER2 model used in the old `_old/semantic_projection/` pipeline. SPECTER2 is trained on scientific paper titles and abstracts — it is domain-specific and produces poor representations for food text. MiniLM is a general-purpose sentence similarity model and produces much more meaningful embeddings for recipe names, descriptions, and ingredients.

---

## Recipe text for embedding

The embedding template (`embed_config.json`) defaults to `name + description + ingredients`. Steps were excluded because they are mostly procedural instructions ("stir until combined", "bake at 350") that add noise without contributing cuisine or category signal. Tags were excluded to avoid leaking the ground truth labels into the embedding.

---

## Cuisine classification: KNN over cosine similarity to label embeddings

The first approach was to embed each cuisine name ("Italian", "Mexican", etc.) and assign each recipe to the nearest cuisine by cosine similarity. This failed badly — "Turkish" accounted for 20% of assignments because the word "Turkish" in MiniLM embedding space is close to "turkey" (the meat), not Turkish cuisine. Recipes involving turkey as an ingredient were being assigned to Turkish cuisine en masse. "French" had a similar problem with French fries and French toast. Single words in a general-purpose embedding model do not reliably represent culinary concepts. "Italian cuisine" and similar phrases were slightly better but still unreliable.

The current approach uses k-nearest-neighbor classification (k=100, distance-weighted, cosine metric) seeded by recipes that already have cuisine tags in the dataset. The KNN classifier learns what Italian, Mexican, etc. actually look like in embedding space from real examples rather than from a word. This is significantly more reliable.

---

## KNN prior correction

KNN posteriors bake in the training class distribution as an implicit prior. American has ~34% of all tagged recipes, so without correction the KNN assigns American to a disproportionate number of ambiguous recipes. Dividing probabilities by class frequencies converts the posterior to a likelihood: P(x | class) rather than P(class | x). This removes the imbalance artifact. The correction is on by default and can be disabled with `--no-prior-correction`.

---

## Leave-one-out probabilities for tagged recipes

When predicting on a recipe that is also a training point, the KNN finds itself as the nearest neighbor (distance = 0) and gives that recipe a probability of 1.0 for its own class. This is technically correct for classification but too confident for the UMAP augmentation step — all tagged Italian recipes would be forced to exactly the same point in the augmented space.

For the probability matrix saved for UMAP (`*_proba.npy`), tagged recipes instead get leave-one-out probabilities: k+1 neighbors are found, the self-match is dropped, and probabilities are computed from the remaining k neighbors. This gives tagged recipes a smooth probability distribution reflecting their actual neighborhood rather than a hard spike.

---

## Cuisine taxonomy from co-occurrence data

The parent-child relationships between cuisine tags (e.g. `cajun` is always a subset of `southern-united-states`) are derived automatically from co-occurrence in the dataset rather than hardcoded. Any tag that appears on 99%+ of the same recipes as another tag, and has fewer total recipes, is treated as a child. This gives us the hierarchy for free from the data and avoids manual maintenance.

When a recipe has multiple matching cuisine tags, the most specific (deepest) tag is used as the training label. A recipe tagged both `american` and `southern-united-states` trains as `southern-united-states`.

---

## UMAP augmentation with category probabilities

The raw 384-dim embeddings capture semantic similarity of recipe text but do not inherently cluster by cuisine, meal type, or other categorical dimensions. To pull same-category recipes together in the 3D projection, we concatenate the KNN probability vectors from each `assign.py` run to the embeddings before UMAP:

```
augmented = concat(embedding, weight * proba_cuisines, weight * proba_meal_types, ...)
```

Using the full probability vector (not just the winning class) means ambiguous recipes sit between their candidate clusters rather than being forced into one. The `--category-weight` parameter controls how strongly the categorical signal influences the projection relative to the semantic embedding.

---

## Meal type classification: type-based only

The initial meal type list included temporal categories (breakfast, lunch, snacks) alongside type categories (main-dish, desserts, side-dishes). This caused high ambiguity: 12% of recipes were tagged with multiple meal types because `lunch + main-dish`, `snacks + appetizers`, etc. co-occur naturally. "Lunch" describes when you eat it; "main-dish" describes what it is.

We dropped the temporal categories and kept only type-based ones: `main-dish`, `desserts`, `side-dishes`, `appetizers`, `beverages`. This reduced ambiguity to 5% and gave cleaner training labels.

---

## Soft binning for numeric features

For fields like `n_steps` and `minutes`, hard one-hot bins create sharp discontinuities: a recipe with 10 steps and one with 11 steps would be maximally different despite being nearly identical. Soft binning uses linear interpolation between bin centers so that values between two centers get fractional weight on both. This gives a smooth representation that better reflects the underlying continuous variable.

For dates (`submitted`), we use min-max normalization to [0, 1] instead of bins. The bin centers for a date range would be arbitrary choices, whereas normalization just captures older vs. newer as a continuous signal.

---

## Tag-based vs. KNN-based encoding

Two approaches exist for adding category information to the UMAP:

- `assign.py` (KNN): learns from tagged training examples, classifies ALL recipes including untagged ones. Use this when you want inferred categories for the whole dataset and the tagging is incomplete.
- `tag_encode.py` (direct lookup): assigns based on tag presence only, untagged recipes get a catch-all label. Use this when the tag is objective and you do not want inference (e.g. cooking time, dietary flags).

`tag_encode.py` is appropriate where false positives are costly. A KNN classifier might infer that a beef recipe is vegetarian based on its neighbors; direct tag lookup will never do that.

---

## Separate scripts for each pipeline stage

The pipeline is split into independent scripts (embed, assign, project, tag_encode, numeric_encode) rather than one monolithic script. Each step writes files that the next step reads. This means:

- Any step can be rerun without redoing the expensive steps before it
- New category assignments can be added without touching the embeddings or projection
- UMAP can be re-projected with different parameters or a different set of category files without re-embedding

The tradeoff is more files on disk, but the cost is low given that outputs are gitignored.

---

## Consistent JSON output across all encoding scripts

All encoding scripts (`assign.py`, `tag_encode.py`, `numeric_encode.py`) output a per-recipe JSON file in the same format:

```json
[{"id": 137739, "category": "mexican", "score": 0.72, "runners_up": [...]}]
```

This means `preview.py` only needs to know about one format regardless of how the categories were produced. The `.npy` files exist solely for UMAP augmentation in `project.py`. `numeric_encode.py` with `type: normalize` skips the JSON output because a continuous normalized value is not a categorical assignment and cannot be meaningfully colored in the preview.

---

## Cook time: numeric minutes over tag-based bins

Initially cook time was encoded via `tag_encode.py` using tags like `15-minutes-or-less`. This was replaced with `numeric_encode.py` using the raw `minutes` field with soft bins. The raw numeric value is more precise and doesn't depend on whether the recipe contributor remembered to add the tag. Outliers (e.g. multi-day fermentation recipes claiming 4000+ minutes) are handled naturally by soft binning — they all get full weight on the last bin.

---

## Minutes and n_steps collinearity

The `minutes` and `n_steps` features are correlated — more steps generally means more time. Including both in the UMAP augmentation doubles the "recipe complexity" signal and causes that dimension to dominate the projection. `minutes` also has more severe outliers than `n_steps`. The recommended approach is to use only one of the two, or to set a very low weight for `minutes` in `projection_weights.json`.

---

## Per-file projection weights

Different category signals have different levels of relevance and different natural scales. A single global weight applied to all proba/feature vectors causes whichever signal has the most variance to dominate the projection. `projection_weights.json` allows tuning each input file independently. The key is the filename stem without the `recipes_` prefix and `.npy` extension (e.g. `"cuisines_proba": 1.0`). A `"default"` key covers any file not explicitly listed.

---

## Preview HTML file size and max-rows

The preview generates a static standalone HTML file. With one trace per category label per family (required for legend click-to-filter to work in plotly), every family's traces are embedded in the HTML even when hidden. Hover text is the dominant cost — at ~300 chars per point, 231k points across 4 families produces a ~500 MB file that browsers cannot load.

The fix is `--max-rows` (default 50k), which keeps the file around 65 MB. At 50k points the structure of the projection is still clearly visible. Run with a higher value if you need to inspect specific sparse regions, but expect slower load times above 100k points.

---

## Preview: one trace per label, not one trace per family

An earlier version of the preview used a single scatter trace per category family with the color array swapped via the dropdown. This kept the HTML small but broke plotly's built-in legend click behavior — clicking a legend entry only toggled the empty dummy trace, not the actual data points.

The current approach uses one trace per category label per family. This is the only way to get native legend click-to-filter behavior in plotly without custom JavaScript. The file size cost is managed with `--max-rows`.
