import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import ast
import re
import os
import kagglehub

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food.com Explorer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* Root theme */
:root {
    --cream:   #fdf6ee;
    --rust:    #c0392b;
    --amber:   #e67e22;
    --forest:  #27ae60;
    --ink:     #1a1a2e;
    --muted:   #7f8c8d;
    --card-bg: #ffffff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--ink);
}

/* Header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #c0392b 60%, #e67e22 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.1rem;
    opacity: 0.85;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 5px solid var(--rust);
    margin-bottom: 1rem;
}
.metric-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--rust);
    line-height: 1;
}
.metric-card .label {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--ink);
    border-bottom: 3px solid var(--amber);
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--ink) !important;
}
[data-testid="stSidebar"] * {
    color: #ecf0f1 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 1rem;
    padding: 0.3rem 0;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

/* Hide default Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading dataset from Kaggle…")
def load_data():
    dl_path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
    interactions = pd.read_csv(os.path.join(dl_path, "RAW_interactions.csv"), parse_dates=["date"])
    recipes      = pd.read_csv(os.path.join(dl_path, "RAW_recipes.csv"),      parse_dates=["submitted"])
    return interactions, recipes


@st.cache_data(show_spinner="Merging tables…")
def merge_data(_interactions, _recipes):
    merged = _interactions.merge(
        _recipes[["id", "name", "minutes", "n_ingredients", "n_steps",
                  "ingredients", "tags", "submitted", "nutrition"]],
        left_on="recipe_id", right_on="id", how="left"
    )
    return merged


@st.cache_data(show_spinner="Parsing ingredients…")
def parse_ingredients(_recipes):
    """Flatten the ingredients lists into a single Counter."""
    all_ingr = []
    for row in _recipes["ingredients"].dropna():
        try:
            items = ast.literal_eval(row)
            all_ingr.extend([i.strip().lower() for i in items])
        except Exception:
            pass
    return Counter(all_ingr)


@st.cache_data(show_spinner="Computing popularity stats…")
def compute_popularity(_merged):
    pop = (
        _merged.groupby(["recipe_id", "name"])
        .agg(avg_rating=("rating", "mean"),
             num_reviews=("rating", "count"),
             rating_std=("rating", "std"))
        .reset_index()
        .dropna(subset=["name"])
    )
    pop["avg_rating"] = pop["avg_rating"].round(2)
    return pop


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Food.com Explorer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Overview", "⭐  Recipe Popularity", "🧄  Ingredient Trends"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#95a5a6'>Upload your CSVs in the same folder as app.py</small>",
        unsafe_allow_html=True,
    )


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    interactions, recipes = load_data()
except FileNotFoundError as e:
    st.error(
        f"⚠️  Could not find data file: **{e.filename}**\n\n"
        "Please place `RAW_interactions.csv` and `RAW_recipes.csv` "
        "in the **same folder** as `app.py`, then refresh."
    )
    st.stop()

merged      = merge_data(interactions, recipes)
ingr_counter = parse_ingredients(recipes)
popularity  = compute_popularity(merged)

# ── COLOUR PALETTE for charts ─────────────────────────────────────────────────
PALETTE = ["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad",
           "#d35400", "#16a085", "#f39c12", "#2c3e50", "#e74c3c"]
RUST   = "#c0392b"
AMBER  = "#e67e22"
FOREST = "#27ae60"


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div class="hero">
        <h1>🍽️ Food.com Explorer</h1>
        <p>Discover recipe trends, popularity patterns, and ingredient insights
           from 230 000+ recipes and 1.1 million user interactions.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, f"{len(recipes):,}",      "Total Recipes",      RUST),
        (c2, f"{len(interactions):,}", "User Interactions",  AMBER),
        (c3, f"{interactions['user_id'].nunique():,}", "Unique Users", FOREST),
        (c4, f"{interactions['rating'].mean():.2f} ★", "Avg Rating", "#2980b9"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="val" style="color:{color}">{val}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Ratings over time + Rating distribution side by side
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-title">Interactions Over Time</div>', unsafe_allow_html=True)
        timeline = (
            merged.dropna(subset=["date"])
            .set_index("date")
            .resample("M")["rating"]
            .count()
            .reset_index()
            .rename(columns={"rating": "interactions", "date": "month"})
        )
        fig = px.area(
            timeline, x="month", y="interactions",
            color_discrete_sequence=[RUST],
            labels={"interactions": "# Interactions", "month": ""},
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0, r=0, t=10, b=0), height=280,
        )
        fig.update_traces(line_color=RUST, fillcolor="rgba(192,57,43,0.15)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Rating Distribution</div>', unsafe_allow_html=True)
        rating_counts = interactions["rating"].value_counts().sort_index()
        fig2 = px.bar(
            x=rating_counts.index, y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"},
            color=rating_counts.index.astype(str),
            color_discrete_sequence=PALETTE,
        )
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Recipe complexity
    st.markdown('<div class="section-title">Recipe Complexity Snapshot</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig3 = px.histogram(
            recipes[recipes["n_steps"] < 40], x="n_steps", nbins=30,
            color_discrete_sequence=[AMBER],
            labels={"n_steps": "Number of Steps"},
            title="Steps per Recipe",
        )
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=30, b=0), height=260)
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        fig4 = px.histogram(
            recipes[recipes["n_ingredients"] < 30], x="n_ingredients", nbins=25,
            color_discrete_sequence=[FOREST],
            labels={"n_ingredients": "Number of Ingredients"},
            title="Ingredients per Recipe",
        )
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=30, b=0), height=260)
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RECIPE POPULARITY
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⭐  Recipe Popularity":
    st.markdown('<div class="section-title">⭐ Recipe Popularity</div>',
                unsafe_allow_html=True)

    # Controls
    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        min_reviews = st.slider("Min. number of reviews", 5, 500, 50)
    with col_b:
        top_n = st.slider("Show top N recipes", 10, 50, 20)
    with col_c:
        sort_by = st.selectbox("Rank by", ["avg_rating", "num_reviews"])

    filtered = popularity[popularity["num_reviews"] >= min_reviews].copy()
    top = filtered.nlargest(top_n, sort_by)

    tab1, tab2, tab3 = st.tabs(["🏆 Top Recipes", "📊 Rating vs Reviews", "📅 Trends Over Time"])

    # ── Tab 1: horizontal bar chart ──────────────────────────────────────────
    with tab1:
        fig = px.bar(
            top.sort_values(sort_by),
            x=sort_by, y="name",
            orientation="h",
            color=sort_by,
            color_continuous_scale=["#f9e4b7", RUST],
            labels={"avg_rating": "Avg Rating", "num_reviews": "# Reviews", "name": ""},
            hover_data={"avg_rating": True, "num_reviews": True},
            height=max(400, top_n * 22),
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False, margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: scatter ───────────────────────────────────────────────────────
    with tab2:
        scatter_data = filtered.copy()
        fig2 = px.scatter(
            scatter_data,
            x="num_reviews", y="avg_rating",
            size="num_reviews",
            hover_name="name",
            color="avg_rating",
            color_continuous_scale=["#f9e4b7", RUST],
            labels={"num_reviews": "Number of Reviews",
                    "avg_rating": "Average Rating"},
            opacity=0.7,
            log_x=True,
        )
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=20, b=0), height=450,
        )
        # Highlight top-10
        top10 = filtered.nlargest(10, "num_reviews")
        fig2.add_traces(px.scatter(
            top10, x="num_reviews", y="avg_rating",
            text="name", color_discrete_sequence=[RUST]
        ).update_traces(
            textposition="top center", textfont_size=9, marker_size=10
        ).data)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: monthly avg rating ────────────────────────────────────────────
    with tab3:
        monthly = (
            merged.dropna(subset=["date"])
            .set_index("date")
            .resample("M")["rating"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_rating", "count": "interactions"})
        )
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(
            go.Scatter(x=monthly["date"], y=monthly["avg_rating"],
                       name="Avg Rating", line=dict(color=RUST, width=2.5)),
            secondary_y=False,
        )
        fig3.add_trace(
            go.Bar(x=monthly["date"], y=monthly["interactions"],
                   name="Interactions", marker_color="rgba(230,126,34,0.25)"),
            secondary_y=True,
        )
        fig3.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=0, r=0, t=30, b=0), height=420,
        )
        fig3.update_yaxes(title_text="Avg Rating", range=[3.5, 5.2], secondary_y=False)
        fig3.update_yaxes(title_text="# Interactions", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Recipe search box ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Search a Recipe")
    query = st.text_input("Type a recipe name…", placeholder="e.g. chocolate chip cookies")
    if query:
        results = popularity[
            popularity["name"].str.contains(query, case=False, na=False)
        ].nlargest(10, "num_reviews")
        if results.empty:
            st.info("No recipes found.")
        else:
            st.dataframe(
                results[["name", "avg_rating", "num_reviews"]]
                .rename(columns={"name": "Recipe", "avg_rating": "Avg ★",
                                  "num_reviews": "Reviews"}),
                use_container_width=True, hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — INGREDIENT TRENDS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧄  Ingredient Trends":
    st.markdown('<div class="section-title">🧄 Ingredient Trends</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔝 Most Common", "📈 Co-occurrence", "⭐ By Rating"])

    # ── Tab 1: top ingredients bar chart ────────────────────────────────────
    with tab1:
        top_n_ingr = st.slider("Show top N ingredients", 10, 60, 30, key="ingr_n")

        # Stopwords to exclude
        stopwords = {"to taste", "to serve", "optional", "garnish"}
        ingr_df = pd.DataFrame(
            [(k, v) for k, v in ingr_counter.items() if k not in stopwords],
            columns=["ingredient", "count"]
        ).nlargest(top_n_ingr, "count")

        fig = px.bar(
            ingr_df.sort_values("count"),
            x="count", y="ingredient", orientation="h",
            color="count",
            color_continuous_scale=["#fde8c8", RUST],
            labels={"count": "Recipe Count", "ingredient": ""},
            height=max(400, top_n_ingr * 22),
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: ingredient co-occurrence heatmap ──────────────────────────────
    with tab2:
        st.markdown(
            "Which ingredients appear **together** most often in the same recipe?"
        )
        top_co = st.slider("Top N ingredients for co-occurrence", 10, 30, 15, key="co_n")
        top_ingr_list = [k for k, _ in ingr_counter.most_common(top_co)]

        # Build co-occurrence matrix
        co_matrix = pd.DataFrame(0, index=top_ingr_list, columns=top_ingr_list)
        for row in recipes["ingredients"].dropna():
            try:
                items = [i.strip().lower() for i in ast.literal_eval(row)]
                present = [i for i in items if i in top_ingr_list]
                for i in present:
                    for j in present:
                        if i != j:
                            co_matrix.loc[i, j] += 1
            except Exception:
                pass

        fig2 = px.imshow(
            co_matrix,
            color_continuous_scale=["white", "#f9c784", RUST],
            aspect="auto",
            labels=dict(color="Co-occurrences"),
        )
        fig2.update_layout(
            margin=dict(l=0, r=0, t=20, b=0), height=520,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: avg rating per ingredient ────────────────────────────────────
    with tab3:
        st.markdown(
            "Average recipe rating for the **top 25 most-used ingredients**."
        )
        top25 = [k for k, _ in ingr_counter.most_common(25)]

        ingr_rating = []
        for _, row in merged.dropna(subset=["ingredients", "rating"]).iterrows():
            try:
                items = [i.strip().lower() for i in ast.literal_eval(row["ingredients"])]
                for ingr in items:
                    if ingr in top25:
                        ingr_rating.append({"ingredient": ingr, "rating": row["rating"]})
            except Exception:
                pass

        if ingr_rating:
            ir_df = (
                pd.DataFrame(ingr_rating)
                .groupby("ingredient")
                .agg(avg_rating=("rating", "mean"), count=("rating", "count"))
                .reset_index()
                .sort_values("avg_rating", ascending=False)
            )
            fig3 = px.bar(
                ir_df,
                x="avg_rating", y="ingredient", orientation="h",
                color="avg_rating",
                color_continuous_scale=["#fde8c8", FOREST],
                labels={"avg_rating": "Avg Recipe Rating", "ingredient": ""},
                hover_data={"count": True},
                height=560,
            )
            fig3.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(tickfont=dict(size=11)),
                xaxis=dict(range=[3.5, 5]),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough merged data to compute per-ingredient ratings.")
