import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-banner h1 {
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 .4rem;
        letter-spacing: -1px;
    }
    .hero-banner p {
        color: #a89fcb;
        font-size: 1.05rem;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 999px;
        padding: .2rem .85rem;
        font-size: .78rem;
        color: #d4cdf7;
        margin-bottom: .9rem;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    /* metric cards */
    .metric-row { display: flex; gap: 14px; margin-bottom: 1.4rem; }
    .metric-card {
        flex: 1;
        background: #1a1a2e;
        border: 1px solid #2e2e5e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    .metric-card .label {
        font-size: .75rem;
        color: #7a70a8;
        text-transform: uppercase;
        letter-spacing: .6px;
        margin-bottom: .3rem;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #e0d7ff;
    }

    /* recommendation card */
    .rec-card {
        background: linear-gradient(145deg, #1e1e3a, #16162a);
        border: 1px solid #2e2e5e;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: .75rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: border-color .2s;
    }
    .rec-card:hover { border-color: #6c63c7; }
    .rec-rank {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6c63c7;
        min-width: 2rem;
        text-align: center;
    }
    .rec-info { flex: 1; }
    .rec-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f0ecff;
        margin: 0 0 .25rem;
    }
    .rec-genre {
        font-size: .82rem;
        color: #7a70a8;
    }
    .rec-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-top: .4rem; }
    .badge {
        display: inline-block;
        border-radius: 999px;
        font-size: .73rem;
        font-weight: 500;
        padding: .15rem .65rem;
    }
    .badge-rating {
        background: #2b2050;
        color: #c4b8ff;
        border: 1px solid #4a3f8a;
    }
    .badge-sim {
        background: #1a2e20;
        color: #7de8a0;
        border: 1px solid #2e5e40;
    }
    .badge-genre {
        background: #2e1a28;
        color: #e89fd4;
        border: 1px solid #5e2e50;
    }

    /* sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0f0e1a;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] p {
        color: #c0b8e0 !important;
    }

    /* info box */
    .info-box {
        background: #12121f;
        border: 1px solid #2e2e5e;
        border-left: 3px solid #6c63c7;
        border-radius: 10px;
        padding: .9rem 1.1rem;
        font-size: .88rem;
        color: #9990c0;
        margin-top: 1rem;
    }

    /* section header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c8c0ff;
        margin: 1.5rem 0 .8rem;
        padding-bottom: .4rem;
        border-bottom: 1px solid #2e2e5e;
    }

    /* genre pill cloud */
    .pill-cloud { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.2rem; }
    .genre-pill {
        background: #1e1a3a;
        border: 1px solid #3e3570;
        border-radius: 999px;
        padding: .3rem .85rem;
        font-size: .8rem;
        color: #b0a8e0;
    }
    .genre-pill span {
        background: #3e3570;
        border-radius: 999px;
        padding: .05rem .45rem;
        font-size: .7rem;
        color: #d0c8ff;
        margin-left: .35rem;
    }

    /* hide default streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data ───────────────────────────────────────────────────────────────────────
def load_movies() -> pd.DataFrame:
    data = {
        "MovieID": range(1, 27),
        "Title": [
            "The Dark Knight", "Inception", "Interstellar", "The Matrix",
            "Avengers: Endgame", "Iron Man", "Spider-Man", "Thor",
            "Titanic", "The Notebook", "La La Land", "Pride & Prejudice",
            "The Shawshank Redemption", "Forrest Gump", "Schindler's List",
            "The Godfather", "Toy Story", "Finding Nemo", "The Lion King",
            "Shrek", "Get Out", "A Quiet Place", "The Conjuring", "It",
            "Parasite", "Oppenheimer",
        ],
        "Genre": [
            "Action Thriller Crime", "Action SciFi Thriller",
            "SciFi Drama Adventure", "Action SciFi",
            "Action Adventure SciFi", "Action SciFi Adventure",
            "Action Adventure", "Action Adventure Fantasy",
            "Romance Drama", "Romance Drama", "Romance Drama Music",
            "Romance Drama", "Drama Crime", "Drama Romance Comedy",
            "Drama History War", "Crime Drama",
            "Animation Comedy Adventure", "Animation Comedy Adventure",
            "Animation Drama Adventure", "Animation Comedy Fantasy",
            "Horror Thriller Mystery", "Horror SciFi Thriller",
            "Horror Mystery", "Horror", "Drama Thriller Crime",
            "Drama History Biography",
        ],
        "Year": [
            2008, 2010, 2014, 1999, 2019, 2008, 2002, 2011,
            1997, 2004, 2016, 2005, 1994, 1994, 1993, 1972,
            1995, 2003, 1994, 2001, 2017, 2018, 2013, 2017, 2019, 2023,
        ],
        "Rating": [
            9.0, 8.8, 8.6, 8.7, 8.4, 7.9, 7.4, 7.0,
            7.8, 7.9, 8.0, 7.8, 9.3, 8.8, 9.0, 9.2,
            8.3, 8.2, 8.5, 7.9, 7.7, 7.5, 7.5, 6.9, 8.5, 8.9,
        ],
        "Description": [
            "Batman fights the Joker in Gotham city dark hero vigilante",
            "Dream thief enters subconscious mind layers reality illusion",
            "Astronauts travel wormhole space time gravity black hole",
            "Hacker discovers reality is simulation artificial intelligence machines",
            "Superheroes assemble fight Thanos save universe infinity stones",
            "Billionaire builds suit armor becomes superhero technology",
            "Teenager bitten spider gains powers protects New York city",
            "Norse god banished Earth hammer lightning Asgard Loki",
            "Love story ship iceberg tragedy ocean Rose Jack",
            "Small town love story second chances heartbreak reunion",
            "Jazz musician actress dreams Los Angeles music ambition",
            "19th century England romance wit society marriage Bennet Darcy",
            "Prison friendship hope freedom wrongful conviction escape",
            "Simple man extraordinary journey American history shrimp war",
            "Holocaust rescue Jewish factory Poland World War II",
            "Mafia family power crime loyalty Corleone Sicily",
            "Toys come alive adventure friendship loyalty cowboy space ranger",
            "Clownfish ocean search son Great Barrier Reef turtle",
            "Lion cub exile return pride kingdom uncle betrayal",
            "Ogre swamp fairy tale princess Dragon talking donkey",
            "Black man sunken place racism horror thriller psychological",
            "Family hides from blind creatures silence survival alien",
            "Paranormal investigators haunted house demons possession",
            "Clown sewer children fear Pennywise shapeshifter",
            "Class inequality dark comedy parasite rich poor family",
            "Physicist J Robert Oppenheimer atomic bomb Manhattan Project nuclear weapons war guilt moral",
        ],
    }
    return pd.DataFrame(data)


@st.cache_resource
def prepare_engine():
    movies = load_movies().copy()
    movies["combined"] = movies["Genre"] + " " + movies["Description"]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(movies["combined"])
    sim = cosine_similarity(mat, mat)
    return movies, sim


def get_recommendations(title, movies, sim, n=5):
    idx = movies[movies["Title"] == title].index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1: n + 1]
    rec_idx = [x[0] for x in scores]
    result = movies.iloc[rec_idx][["Title", "Genre", "Year", "Rating"]].copy()
    result["Similarity"] = [round(x[1] * 100, 1) for x in scores]
    result.index = range(1, len(result) + 1)
    return result


# ── Load data ──────────────────────────────────────────────────────────────────
movies_df, sim_matrix = prepare_engine()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("---")

    selected_movie = st.selectbox(
        "🎥 Pick a movie you liked",
        sorted(movies_df["Title"].tolist()),
        help="Select any movie to find similar ones.",
    )

    top_n = st.slider(
        "🔢 Number of recommendations",
        min_value=3, max_value=10, value=5,
        help="How many similar movies to display.",
    )

    st.markdown("---")

    # Genre filter
    all_genres = sorted(
        set(g for genres in movies_df["Genre"].str.split() for g in genres)
    )
    genre_filter = st.multiselect(
        "🏷️ Filter catalogue by genre",
        options=all_genres,
        default=[],
        help="Leave empty to show all genres.",
    )

    st.markdown("---")
    st.markdown(
        "<div style='color:#7a70a8;font-size:.8rem;'>Built with TF-IDF & Cosine Similarity · "
        "<a href='https://github.com/venkatasriharika-code/movie_recommendation_system' "
        "style='color:#a89fcb;'>GitHub ↗</a></div>",
        unsafe_allow_html=True,
    )

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">Content-Based Filtering</div>
        <h1>🎬 CineMatch</h1>
        <p>Discover movies you'll love · Powered by TF-IDF & Cosine Similarity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Top-level metrics ──────────────────────────────────────────────────────────
avg_rating = movies_df["Rating"].mean()
top_rated = movies_df.loc[movies_df["Rating"].idxmax(), "Title"]
unique_genres = len(set(g for gs in movies_df["Genre"].str.split() for g in gs))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎞️ Movies in Catalogue", len(movies_df))
with col2:
    st.metric("⭐ Avg IMDb Rating", f"{avg_rating:.2f}")
with col3:
    st.metric("🏷️ Unique Genres", unique_genres)
with col4:
    st.metric("🥇 Top Rated", top_rated[:18] + ("…" if len(top_rated) > 18 else ""))

st.divider()

# ── Main content ───────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

# ── LEFT: Recommendations ──────────────────────────────────────────────────────
with left:
    st.markdown(f"### 🎯 Movies Similar to *{selected_movie}*")

    recs = get_recommendations(selected_movie, movies_df, sim_matrix, top_n)

    # Selected movie info
    sel = movies_df[movies_df["Title"] == selected_movie].iloc[0]
    with st.container():
        st.info(
            f"**{sel['Title']}** ({sel['Year']})  ·  ⭐ {sel['Rating']}  ·  🏷️ {sel['Genre']}"
        )

    # Recommendation cards
    for rank, row in recs.iterrows():
        genre_tags = " · ".join(row["Genre"].split())
        sim_color = "🟢" if row["Similarity"] >= 50 else ("🟡" if row["Similarity"] >= 30 else "🔴")

        st.markdown(
            f"""
            <div class="rec-card">
                <div class="rec-rank">#{rank}</div>
                <div class="rec-info">
                    <div class="rec-title">{row['Title']} <span style="font-size:.82rem;font-weight:400;color:#7a70a8">({row['Year']})</span></div>
                    <div class="rec-genre">{genre_tags}</div>
                    <div class="rec-badges">
                        <span class="badge badge-rating">⭐ {row['Rating']}</span>
                        <span class="badge badge-sim">{sim_color} {row['Similarity']}% match</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Download CSV
    csv = recs.reset_index().rename(columns={"index": "Rank"}).to_csv(index=False)
    st.download_button(
        "⬇️ Download Recommendations as CSV",
        data=csv,
        file_name=f"{selected_movie.replace(' ', '_')}_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── RIGHT: Catalogue & Stats ───────────────────────────────────────────────────
with right:
    st.markdown("### 📊 Catalogue Insights")

    # Genre distribution
    genre_counts = (
        movies_df["Genre"].str.split().explode()
        .value_counts().rename_axis("Genre").reset_index(name="Count")
    )

    st.markdown("**Top genres in catalogue**")
    max_count = genre_counts["Count"].max()
    for _, row in genre_counts.head(8).iterrows():
        pct = row["Count"] / max_count
        bar_w = int(pct * 100)
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <div style="min-width:130px;font-size:.85rem;color:#c8c0ff;">{row['Genre']}</div>
                <div style="flex:1;background:#1e1a3a;border-radius:999px;height:8px;overflow:hidden">
                    <div style="width:{bar_w}%;background:linear-gradient(90deg,#6c63c7,#a89fcb);height:100%;border-radius:999px;"></div>
                </div>
                <div style="font-size:.8rem;color:#7a70a8;min-width:24px;text-align:right">{row['Count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Filtered catalogue
    st.markdown("**Browse catalogue**")

    display_df = movies_df[["Title", "Genre", "Year", "Rating"]].copy()
    if genre_filter:
        mask = display_df["Genre"].apply(
            lambda g: any(gf in g.split() for gf in genre_filter)
        )
        display_df = display_df[mask]

    st.dataframe(
        display_df.sort_values("Rating", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rating": st.column_config.ProgressColumn(
                "Rating", min_value=0, max_value=10, format="%.1f"
            )
        },
    )

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;color:#4a4270;font-size:.82rem;padding:.5rem 0 1.5rem;">
        CineMatch · Content-Based Movie Recommender ·
        <a href="https://github.com/venkatasriharika-code/movie_recommendation_system"
           style="color:#6c63c7;text-decoration:none;">View on GitHub ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)
