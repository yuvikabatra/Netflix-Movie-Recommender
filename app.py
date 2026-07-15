import ast
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from auth import auth_gate
if not auth_gate():
       st.stop()

st.set_page_config(page_title="Netflix Recommendation System", layout="wide", page_icon="🎬")

try:
    API_KEY = st.secrets["api_key"]
except Exception:
    API_KEY = None

SESSION = requests.Session()

# --------------------------------------------------------------------------------------
# THEME / CSS
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0b0f; color: #e5e5e5; }

    section[data-testid="stSidebar"] {
        background-color: #111114;
        border-right: 1px solid #262626;
    }
    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 2px;
        width: 100%;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #E50914;
    }

    .netflix-logo {
        color: #E50914;
        font-weight: 800;
        font-size: 26px;
        letter-spacing: 1px;
        margin-bottom: 0px;
    }
    .netflix-sub {
        color: #9a9a9a;
        font-size: 11px;
        letter-spacing: 2px;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    .metric-card {
        background-color: #16161a;
        border: 1px solid #262626;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: left;
    }
    .metric-label { color: #9a9a9a; font-size: 12px; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; }

    .panel {
        background-color: #16161a;
        border: 1px solid #262626;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 16px;
    }
    .panel-title { font-weight: 700; font-size: 16px; margin-bottom: 10px; }

    .movie-card {
        background-color: #16161a;
        border: 1px solid #262626;
        border-radius: 10px;
        padding: 10px;
        text-align: left;
        height: 100%;
    }
    .movie-title { color: #E50914; font-weight: 700; margin-top: 8px; font-size: 14px; }
    .movie-meta { color: #9a9a9a; font-size: 12px; line-height: 1.5; }
    .movie-match { color: #2ecc71; font-weight: 700; font-size: 13px; margin-top: 6px; }

    div.stButton > button {
        background-color: #E50914;
        color: white;
        border: none;
        font-weight: 600;
    }
    div.stButton > button:hover { background-color: #b80710; color: white; }

    h1, h2, h3 { color: #f2f2f2; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent
    movies = pd.read_csv(base / "tmdb_5000_movies.csv")
    credits = pd.read_csv(base / "tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    keep = [
        "id_x" if "id_x" in movies.columns else "id",
        "title", "overview", "genres", "keywords", "cast", "crew",
        "release_date", "vote_average", "vote_count",
        "production_countries", "original_language",
    ]
    movies = movies[keep].rename(columns={keep[0]: "movie_id"}).dropna(subset=["title", "overview"])

    def parse_names(x, limit=None):
        try:
            items = [i["name"] for i in ast.literal_eval(x)]
            return items[:limit] if limit else items
        except Exception:
            return []

    def parse_director(x):
        try:
            for i in ast.literal_eval(x):
                if i["job"] == "Director":
                    return [i["name"]]
        except Exception:
            pass
        return []

    movies["genres_list"] = movies["genres"].apply(parse_names)
    movies["keywords_list"] = movies["keywords"].apply(lambda x: parse_names(x))
    movies["cast_list"] = movies["cast"].apply(lambda x: parse_names(x, 3))
    movies["crew_list"] = movies["crew"].apply(parse_director)
    movies["countries_list"] = movies["production_countries"].apply(parse_names)

    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["year"] = movies["release_date"].dt.year
    movies["overview_tokens"] = movies["overview"].fillna("").apply(lambda x: x.split())

    tag_source = pd.DataFrame({
        "overview": movies["overview_tokens"],
        "genres": movies["genres_list"].apply(lambda l: [i.replace(" ", "") for i in l]),
        "keywords": movies["keywords_list"].apply(lambda l: [i.replace(" ", "") for i in l]),
        "cast": movies["cast_list"].apply(lambda l: [i.replace(" ", "") for i in l]),
        "crew": movies["crew_list"].apply(lambda l: [i.replace(" ", "") for i in l]),
    })
    movies["tags"] = (
        tag_source["overview"] + tag_source["genres"] + tag_source["keywords"]
        + tag_source["cast"] + tag_source["crew"]
    ).apply(lambda x: " ".join(x).lower())

    movies["genres_display"] = movies["genres_list"].apply(lambda l: ", ".join(l) if l else "Unknown")
    movies["country_display"] = movies["countries_list"].apply(lambda l: l[0] if l else "Unknown")

    return movies.reset_index(drop=True)


@st.cache_data
def build_model():
    m = load_data()
    vec = CountVectorizer(max_features=5000, stop_words="english").fit_transform(m["tags"]).toarray()
    sim = cosine_similarity(vec)
    return m, sim


@st.cache_data
def get_poster(movie_id):
    if not API_KEY:
        return None
    try:
        r = SESSION.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        p = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{p}" if p else None
    except Exception:
        return None


def recommend(title, n=5):
    m, sim = build_model()
    if title not in m["title"].values:
        return []
    idx = m[m.title == title].index[0]
    ranked = sorted(list(enumerate(sim[idx])), key=lambda x: x[1], reverse=True)[1 : n + 1]
    out = []
    for i, score in ranked:
        row = m.iloc[i]
        out.append({
            "title": row["title"],
            "poster": get_poster(int(row["movie_id"])),
            "genres": row["genres_display"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "N/A",
            "rating": row["vote_average"],
            "country": row["country_display"],
            "match": round(score * 100),
        })
    return out


movies, _ = build_model()

# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="netflix-logo">NETFLIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="netflix-sub">RECOMMENDATION SYSTEM</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠 Home", "📊 Data Overview", "🔍 Search & Recommend", "📈 Visualizations", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filters**")

    all_genres = sorted({g for row in movies["genres_list"] for g in row})
    genre_filter = st.selectbox("Select Genre", ["All"] + all_genres)

    rating_filter = st.selectbox("Select Rating", ["All", "9+", "8+", "7+", "6+", "Below 6"])

    min_year = int(movies["year"].min(skipna=True)) if movies["year"].notna().any() else 1950
    max_year = int(movies["year"].max(skipna=True)) if movies["year"].notna().any() else 2024
    year_range = st.slider("Release Year Range", min_year, max_year, (min_year, max_year))

    if st.button("↺ Clear Filters"):
        st.rerun()

# apply filters
filtered = movies.copy()
if genre_filter != "All":
    filtered = filtered[filtered["genres_list"].apply(lambda l: genre_filter in l)]
if rating_filter != "All":
    thresholds = {"9+": 9, "8+": 8, "7+": 7, "6+": 6}
    if rating_filter == "Below 6":
        filtered = filtered[filtered["vote_average"] < 6]
    else:
        filtered = filtered[filtered["vote_average"] >= thresholds[rating_filter]]
filtered = filtered[filtered["year"].between(year_range[0], year_range[1]) | filtered["year"].isna()]

# --------------------------------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("# Netflix Recommendation System 🍿")
    st.write("Get movie recommendations based on content similarity")

    total_titles = len(movies)
    total_genres = len(all_genres)
    total_countries = movies["countries_list"].explode().nunique()
    total_languages = movies["original_language"].nunique()
    avg_rating = round(movies["vote_average"].mean(), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, color in zip(
        [c1, c2, c3, c4, c5],
        ["Total Titles", "Genres", "Countries", "Languages", "Avg Rating"],
        [f"{total_titles:,}", total_genres, total_countries, total_languages, avg_rating],
        ["#E50914", "#a855f7", "#22c55e", "#3b82f6", "#f59e0b"],
    ):
        col.markdown(
            f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">{value}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">🎬 Find Your Next Watch</div>', unsafe_allow_html=True)
        title_options = filtered["title"].tolist() or movies["title"].tolist()
        selected_title = st.selectbox("Select a Movie", title_options, label_visibility="collapsed")
        do_recommend = st.button("Recommend →", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        row = movies[movies["title"] == selected_title].iloc[0]
        poster_url = get_poster(int(row["movie_id"]))
        st.markdown('<div class="panel"><div class="panel-title">About This Title</div>', unsafe_allow_html=True)
        c_img, c_txt = st.columns([1, 2])
        with c_img:
            if poster_url:
                st.image(poster_url, use_container_width=True)
        with c_txt:
            st.markdown(f"**:red[{row['title']}]**")
            st.markdown(
                f"""<div class="movie-meta">
                Genre: {row['genres_display']}<br>
                Country: {row['country_display']}<br>
                Release Year: {row['year'] if pd.notna(row['year']) else 'N/A'}<br>
                Rating: {row['vote_average']}<br>
                <br>Description:<br>{row['overview'][:220]}...
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### ⭐ Top 5 Recommendations")
        results = recommend(selected_title, 5)
        cols = st.columns(5)
        for col, r in zip(cols, results):
            with col:
                card = '<div class="movie-card">'
                st.markdown(card, unsafe_allow_html=True)
                if r["poster"]:
                    st.image(r["poster"], use_container_width=True)
                st.markdown(
                    f"""<div class="movie-title">{r['title']}</div>
                    <div class="movie-meta">
                    Genre: {r['genres']}<br>
                    Year: {r['year']}<br>
                    Rating: {r['rating']}<br>
                    Country: {r['country']}
                    </div>
                    <div class="movie-match">{r['match']}% Match</div>""",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        st.markdown("##### Rating Distribution")
        import plotly.express as px
        bins = pd.cut(movies["vote_average"], bins=[0, 4, 6, 7, 8, 10],
                       labels=["<4", "4-6", "6-7", "7-8", "8-10"])
        dist = bins.value_counts().sort_index()
        fig = px.pie(values=dist.values, names=dist.index.astype(str), hole=0.6,
                      color_discrete_sequence=["#E50914", "#a855f7", "#3b82f6", "#22c55e", "#f59e0b"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e5e5e5", showlegend=True, height=300,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("##### Top 10 Genres")
        import plotly.express as px
        genre_counts = movies["genres_list"].explode().value_counts().head(10)
        fig = px.bar(x=genre_counts.index, y=genre_counts.values,
                      color_discrete_sequence=["#a855f7"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e5e5e5", xaxis_title="", yaxis_title="Count",
                           height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with ch3:
        st.markdown("##### Titles Added Over the Years")
        import plotly.express as px
        year_counts = movies.dropna(subset=["year"]).groupby("year").size()
        fig = px.line(x=year_counts.index, y=year_counts.values, markers=True)
        fig.update_traces(line_color="#22c55e", marker_color="#22c55e")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e5e5e5", xaxis_title="", yaxis_title="Count",
                           height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# DATA OVERVIEW PAGE
# --------------------------------------------------------------------------------------
elif page == "📊 Data Overview":
    st.markdown("# 📊 Data Overview")
    st.write(f"Showing {len(filtered):,} of {len(movies):,} titles based on current filters")
    st.dataframe(
        filtered[["title", "genres_display", "year", "vote_average", "country_display"]]
        .rename(columns={
            "title": "Title", "genres_display": "Genres", "year": "Year",
            "vote_average": "Rating", "country_display": "Country",
        }),
        use_container_width=True,
        height=600,
    )

# --------------------------------------------------------------------------------------
# SEARCH & RECOMMEND PAGE
# --------------------------------------------------------------------------------------
elif page == "🔍 Search & Recommend":
    st.markdown("# 🔍 Search & Recommend")
    title_options = filtered["title"].tolist() or movies["title"].tolist()
    sel = st.selectbox("Choose a title", title_options)
    n = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend →"):
        results = recommend(sel, n)
        cols = st.columns(min(len(results), 5) or 1)
        for i, r in enumerate(results):
            with cols[i % len(cols)]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                if r["poster"]:
                    st.image(r["poster"], use_container_width=True)
                st.markdown(
                    f"""<div class="movie-title">{r['title']}</div>
                    <div class="movie-meta">Genre: {r['genres']}<br>Year: {r['year']}<br>Rating: {r['rating']}</div>
                    <div class="movie-match">{r['match']}% Match</div>""",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# VISUALIZATIONS PAGE
# --------------------------------------------------------------------------------------
elif page == "📈 Visualizations":
    st.markdown("# 📈 Visualizations")
    import plotly.express as px

    genre_counts = filtered["genres_list"].explode().value_counts().head(15)
    fig1 = px.bar(x=genre_counts.values, y=genre_counts.index, orientation="h",
                   color_discrete_sequence=["#E50914"], title="Genre Frequency")
    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e5e5")
    st.plotly_chart(fig1, use_container_width=True)

    year_counts = filtered.dropna(subset=["year"]).groupby("year").size()
    fig2 = px.area(x=year_counts.index, y=year_counts.values, title="Titles Released by Year")
    fig2.update_traces(line_color="#22c55e", fillcolor="rgba(34,197,94,0.2)")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e5e5")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(filtered, x="vote_average", nbins=20, title="Rating Distribution",
                          color_discrete_sequence=["#a855f7"])
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e5e5")
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------------------------------------
else:
    st.markdown("# ℹ️ About")
    st.markdown(
        """
        This app recommends movies based on content similarity (overview, genres,
        keywords, cast, and director) using a bag-of-words model and cosine similarity,
        built on the TMDB 5000 Movies dataset.

        **Built with** ❤️ using Streamlit
        """
    )

st.markdown(
    "<br><center style='color:#666;font-size:12px;'>Built with ❤️ using Streamlit</center>",
    unsafe_allow_html=True,
)