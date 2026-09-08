import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re

from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Movie Finder",
    page_icon="🎬",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

    /* ---------- Main App ---------- */

    .stApp {
        background: #0b0f17;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 35px;
        padding-bottom: 50px;
    }


    /* ---------- Header ---------- */

    .title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .title span {
        color: #ff3b5c;
    }

    .subtitle {
        text-align: center;
        color: #929aaa;
        font-size: 18px;
        margin-bottom: 35px;
    }


    /* ---------- Search Area ---------- */

    .search-title {
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    div[data-baseweb="input"] {
        background: #151a24;
        border: 1px solid #303746;
        border-radius: 12px;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1px solid #ff3b5c;
        box-shadow: 0 0 0 1px #ff3b5c;
    }

    div[data-baseweb="input"] input {
        color: white !important;
        font-size: 17px;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #737c8d !important;
    }


    /* ---------- Button ---------- */

    .stButton > button {
        width: 100%;
        height: 46px;
        border-radius: 11px;
        border: none;
        background: #ff3b5c;
        color: white;
        font-size: 16px;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #ff1744;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(255, 59, 92, 0.25);
    }


    /* ---------- Recommendation Heading ---------- */

    .recommend-title {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-top: 45px;
        margin-bottom: 20px;
    }


    /* ---------- Movie Cards ---------- */

    .movie-card {
        background: #151a24;
        border: 1px solid #272e3b;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .movie-rank {
        color: #ff3b5c;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
    }

    .movie-name {
        color: white;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .movie-description {
        color: #858fa1;
        font-size: 14px;
    }

    .movie-year {
        color: #dce1e8;
        background: #202633;
        border-radius: 20px;
        padding: 7px 13px;
        text-align: center;
        font-size: 14px;
        font-weight: 600;
    }


    /* ---------- Empty State ---------- */

    .empty-title {
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: 700;
        margin-top: 50px;
    }

    .empty-text {
        text-align: center;
        color: #858fa1;
        font-size: 16px;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #596273;
        font-size: 13px;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #202632;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("movies.csv")

    return df


@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


@st.cache_resource
def load_index():

    index = faiss.read_index(
        "movie.index"
    )

    return index


df = load_data()

model = load_model()

index = load_index()

movie_titles = df["Title"].fillna("").tolist()


# ==========================================================
# QUERY PREPROCESSING
# ==========================================================

def preprocess_query(query):

    query = query.lower()

    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query


# ==========================================================
# FUZZY TITLE SEARCH
# ==========================================================

def fuzzy_title_search(
    query,
    limit=50
):

    query = preprocess_query(query)

    matches = process.extract(
        query,
        movie_titles,
        scorer=fuzz.WRatio,
        limit=limit
    )

    results = []

    for title, score, idx in matches:

        results.append({
            "Index": idx,
            "Title Score": score / 100
        })

    return pd.DataFrame(results)


# ==========================================================
# SEMANTIC SEARCH
# ==========================================================

def semantic_search(
    query,
    limit=50
):

    query = preprocess_query(query)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        limit
    )

    results = pd.DataFrame({

        "Index": indices[0],

        "Semantic Score": scores[0]

    })

    return results


# ==========================================================
# HYBRID SEARCH
# ==========================================================

def hybrid_search(
    query,
    limit=50
):

    title_results = fuzzy_title_search(
        query,
        limit
    )

    semantic_results = semantic_search(
        query,
        limit
    )

    title_results = title_results[
        ["Index", "Title Score"]
    ]

    semantic_results = semantic_results[
        ["Index", "Semantic Score"]
    ]

    results = pd.merge(
        title_results,
        semantic_results,
        on="Index",
        how="outer"
    )

    results["Title Score"] = (
        results["Title Score"]
        .fillna(0)
    )

    results["Semantic Score"] = (
        results["Semantic Score"]
        .fillna(0)
    )

    results["Final Score"] = (
        0.60 * results["Title Score"]
        +
        0.40 * results["Semantic Score"]
    )

    results = results.sort_values(
        "Final Score",
        ascending=False
    )

    results = results.head(3)

    results["Title"] = (
        df.iloc[
            results["Index"]
        ]["Title"].values
    )

    results["Year"] = (
        df.iloc[
            results["Index"]
        ]["Year"].values
    )

    results = results[
        [
            "Title",
            "Year"
        ]
    ]

    return results.reset_index(
        drop=True
    )


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="title">
        🎬 <span>Movie</span> Finder
    </div>

    <div class="subtitle">
        Discover movies similar to your favorite films
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# SEARCH
# ==========================================================

st.markdown(
    '<div class="search-title">Search for a movie</div>',
    unsafe_allow_html=True
)

query = st.text_input(
    "Movie Search",
    placeholder="Try Bahubali, Pushpa, RRR...",
    label_visibility="collapsed"
)

search = st.button(
    "🔍  Find Similar Movies"
)


# ==========================================================
# RESULTS
# ==========================================================

if search:

    if not query.strip():

        st.warning(
            "Please enter a movie name."
        )

    else:

        with st.spinner(
            "Finding similar movies..."
        ):

            results = hybrid_search(
                query
            )

        st.markdown(
            """
            <div class="recommend-title">
                🎯 Recommended Movies
            </div>
            """,
            unsafe_allow_html=True
        )

        if results.empty:

            st.markdown(
                """
                <div class="empty-title">
                    🍿 No Movies Found
                </div>

                <div class="empty-text">
                    Try searching with another movie name.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            for rank, (_, movie) in enumerate(results.iterrows(), start=1):

                col1, col2, col3 = st.columns(
                    [0.8, 7, 1.5]
                )

                with col1:

                    st.markdown(
                        f"""
                        <div class="movie-rank">
                            #{rank}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        f"""
                        <div class="movie-name">
                            🎬 {movie["Title"]}
                        </div>

                        <div class="movie-description">
                            Recommended using hybrid NLP search
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:

                    st.markdown(
                        f"""
                        <div class="movie-year">
                            📅 {movie["Year"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    "<div style='height:8px'></div>",
                    unsafe_allow_html=True
                )


# ==========================================================
# INITIAL MESSAGE
# ==========================================================

else:

    st.markdown(
        """
        <div class="empty-title">
            🍿 Find your next movie
        </div>

        <div class="empty-text">
            Enter a movie name above and discover
            your top 3 similar movies.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        Powered by SBERT • FAISS • RapidFuzz • Python
    </div>
    """,
    unsafe_allow_html=True
)