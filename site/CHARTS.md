# Chart System

Charts are rendered with D3.js v7. They appear in two places:
- **Bottom-left panel** (`placement: "panel"`) — a floating panel shared across story and explore modes
- **Inline** (`placement: "inline"`) — embedded in the story panel scroll area (best for small charts)

---

## story.json — Content Block Schema

Each story step has a `content` array of blocks. The step-level `colorBy`, `camera`, and `highlight` fields are unchanged.

```json
{
  "colorBy": "avg_rating",
  "camera": { "position": [...], "target": [...] },
  "highlight": "5 stars",
  "content": [
    { "type": "text", "style": "heading", "value": "The 5-Star Split" },
    { "type": "text", "style": "body", "value": "Paragraph text here." },
    { "type": "description", "value": "A callout or annotation — rendered in italic with a left border." },
    {
      "type": "chart",
      "chartType": "histogram",
      "dataFile": "data/charts/ratings.json",
      "xLabel": "Avg Rating",
      "yLabel": "Count",
      "placement": "panel"
    }
  ]
}
```

### Block types

| `type`        | Required fields         | Optional fields                  |
|---------------|-------------------------|----------------------------------|
| `text`        | `value`                 | `style`: `"heading"` \| `"body"` (default) \| `"caption"` |
| `description` | `value`                 | —                                |
| `chart`       | `chartType`, `dataFile` | `xLabel`, `yLabel`, `placement`, and any chart-type-specific fields |

**One `chart` block per step maximum.** If `placement` is omitted it defaults to `"panel"`.

---

## Chart Data File Format

Each `dataFile` is a JSON file with `chartType`, `data`, and optional axis/range fields. Fields in the `story.json` chart block override the data file's defaults.

### Histogram

Two formats are supported — pre-binned (recommended for small files) and raw values.

**Pre-binned** (ordinal x-axis, uses `labels` + `counts`):
```json
{
  "chartType": "histogram",
  "labels": ["no rating", "1 star", "2 stars", "3 stars", "4 stars", "4.5 stars", "4.8 stars", "5 stars"],
  "counts": [5047, 1454, 2354, 13067, 46335, 39507, 13825, 110048],
  "yLabel": "Recipes"
}
```
- `labels` — category names for each bar (x-axis, rotated to avoid overlap)
- `counts` — integer count per bar (y-axis auto-formatted with K/M suffixes)
- `yMax` — optional; inferred from tallest bar if omitted

Add `categoryFamily` to the chart block in `story.json` to color each bar using the corresponding palette color:
```json
{ "type": "chart", "chartType": "histogram", "dataFile": "...", "categoryFamily": "avg_rating" }
```

**Raw values** (linear x-axis, D3 bins the data):
```json
{
  "chartType": "histogram",
  "data": [4.2, 3.8, 5.0, 4.5, 1.0],
  "bins": 20,
  "xLabel": "Avg Rating",
  "yLabel": "Count",
  "xMin": 0,
  "xMax": 5
}
```
- `data` — flat array of numeric values
- `bins` — number of histogram bins (optional, default 20)
- `xMin`, `xMax` — optional; inferred from data if omitted
- `yMax` — optional; inferred from tallest bin if omitted

### Beeswarm

```json
{
  "chartType": "beeswarm",
  "data": [4.2, 3.8, 5.0, 4.5, 1.0],
  "xLabel": "Avg Rating",
  "xMin": 0,
  "xMax": 5,
  "radius": 3
}
```

- `data` — flat array of numeric values; automatically sampled to ≤600 points for performance
- `radius` — dot radius in pixels (optional, default 3)
- `xMin`, `xMax` — optional; inferred from data if omitted
- No y-axis — dots are force-jittered vertically to avoid overlap

### Line chart

```json
{
  "chartType": "line",
  "data": [[2010, 142], [2011, 198], [2012, 253]],
  "xLabel": "Year",
  "yLabel": "Recipes",
  "xMin": 2008,
  "xMax": 2024,
  "yMin": 0
}
```

- `data` — array of `[x, y]` pairs
- `xMin`, `xMax`, `yMin`, `yMax` — optional; inferred from data if omitted

### Scatter plot

```json
{
  "chartType": "scatter",
  "data": [[4.2, 130], [3.8, 45], [5.0, 20]],
  "xLabel": "Rating",
  "yLabel": "Minutes",
  "xMin": 0,
  "xMax": 5,
  "yMin": 0,
  "yMax": 300
}
```

- `data` — array of `[x, y]` pairs
- `xMin`, `xMax`, `yMin`, `yMax` — optional; inferred from data if omitted

---

## Placement

| `placement` | Behavior |
|-------------|----------|
| `"panel"`   | Chart appears in the bottom-left floating panel. Closes automatically when navigating to a step without a chart, or when switching to Explore mode. |
| `"inline"`  | Chart is rendered inside the story panel scroll area, between other content blocks. Best for small supplementary charts. |
