"""
Content-Based Movie Recommendation System — MovieLens Dataset
================================================================
Dataset : MovieLens (ml-latest-small or ml-25m)
          https://grouplens.org/datasets/movielens/
          Expects movies.csv with columns: movieId, title, genres
          (genres are pipe-separated, e.g. "Adventure|Animation|Comedy")
          Optionally uses tags.csv (movieId, tag) for richer descriptions.

Approach: CONTENT-BASED FILTERING
  1. Load movies (+ optional tags)
  2. Build a "content soup" per movie (genres, and tags if available)
  3. Vectorize with TF-IDF
  4. Compute pairwise Cosine Similarity between all movies
  5. Given a movie title, recommend the N most similar movies
  6. Generate a PDF report showing methodology + example recommendations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.lib import colors

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
MOVIES_CSV_PATH = "movies.csv"
TAGS_CSV_PATH = "tags.csv"          # optional, script works without it
TOP_N = 10
SAMPLE_MOVIES = ["Toy Story (1995)", "The Matrix (1999)", "Titanic (1997)"]  # None -> auto-pick 3 sample movies
REPORT_PATH = "Movie_Recommender_Report.pdf"


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data():
    movies = pd.read_csv(MOVIES_CSV_PATH)
    print(f"movies.csv -> {movies.shape[0]} rows, {movies.shape[1]} cols")

    tags = None
    try:
        tags = pd.read_csv(TAGS_CSV_PATH)
        print(f"tags.csv   -> {tags.shape[0]} rows, {tags.shape[1]} cols")
    except FileNotFoundError:
        print("tags.csv not found — continuing with genres only")

    return movies, tags


# ----------------------------------------------------------------------
# 2. CLEAN + BUILD CONTENT
# ----------------------------------------------------------------------
def clean_and_build_content(movies: pd.DataFrame, tags: pd.DataFrame | None) -> pd.DataFrame:
    df = movies.copy()
    df = df.dropna(subset=["title", "genres"])
    df = df[df["genres"] != "(no genres listed)"]
    df = df.reset_index(drop=True)

    # Extract release year from title, e.g. "Toy Story (1995)"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)")

    # genres: "Adventure|Animation|Comedy" -> "Adventure Animation Comedy"
    genres_text = df["genres"].str.replace("|", " ", regex=False)

    if tags is not None:
        tags_agg = (
            tags.dropna(subset=["tag"])
            .groupby("movieId")["tag"]
            .apply(lambda t: " ".join(t.astype(str)))
        )
        df = df.merge(tags_agg.rename("tags"), on="movieId", how="left")
        df["tags"] = df["tags"].fillna("")
        # weight genres x2 relative to free-text tags so genre signal dominates
        df["content"] = (genres_text + " " + genres_text + " " + df["tags"]).str.strip()
    else:
        df["content"] = genres_text

    return df


# ----------------------------------------------------------------------
# 3-4. TF-IDF + COSINE SIMILARITY
# ----------------------------------------------------------------------
def build_similarity_matrix(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z0-9\-]+")
    tfidf_matrix = vectorizer.fit_transform(df["content"])
    print(f"\nTF-IDF matrix: {tfidf_matrix.shape[0]} movies x {tfidf_matrix.shape[1]} terms")

    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix, vectorizer


# ----------------------------------------------------------------------
# 5. RECOMMENDATION FUNCTION
# ----------------------------------------------------------------------
def recommend(title: str, df: pd.DataFrame, sim_matrix: np.ndarray, top_n=TOP_N) -> pd.DataFrame:
    matches = df.index[df["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        # fall back to a partial match
        matches = df.index[df["title"].str.lower().str.contains(title.lower(), regex=False)]
    if len(matches) == 0:
        raise ValueError(f"Movie not found: {title}")

    idx = matches[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    rec_idx = [s[0] for s in scores]
    rec_scores = [round(s[1], 3) for s in scores]

    result = df.iloc[rec_idx][["title", "genres"]].copy()
    result["similarity"] = rec_scores
    return result.reset_index(drop=True)


def pick_sample_movies(df: pd.DataFrame, n=3):
    """Pick n movies with reasonably rich genre tags for a good demo."""
    genre_richness = df["genres"].str.count(r"\|") + 1
    candidates = df[genre_richness >= 3]["title"]
    if len(candidates) < n:
        candidates = df["title"]
    return list(candidates.sample(n, random_state=42))


# ----------------------------------------------------------------------
# 6. VISUALIZATION — similarity heatmap for one sample movie's neighbors
# ----------------------------------------------------------------------
def plot_recommendation_bar(title: str, rec_df: pd.DataFrame, filename: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(rec_df["title"][::-1], rec_df["similarity"][::-1], color="mediumpurple")
    ax.set_xlabel("Cosine Similarity")
    ax.set_title(f"Top {len(rec_df)} Recommendations for: {title}")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_genre_distribution(df: pd.DataFrame, filename: str, top_n=15):
    all_genres = df["genres"].str.split("|").explode()
    counts = all_genres.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color="teal")
    ax.set_title("Most Common Genres in the Catalog")
    ax.set_ylabel("Number of Movies")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


# ----------------------------------------------------------------------
# 7. PDF REPORT
# ----------------------------------------------------------------------
def build_pdf_report(df: pd.DataFrame, sample_results: list):
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=9, leading=12,
                            textColor=colors.grey)

    doc = SimpleDocTemplate(
        REPORT_PATH, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    story = []

    # --- Cover ---
    story.append(Paragraph("Content-Based Movie Recommendation System", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Built on the MovieLens dataset ({df.shape[0]} movies). "
        "Recommendations are generated using TF-IDF vectorization of movie genres "
        "(and tags, if available) followed by cosine similarity between movies — "
        "no user ratings are needed, so this works even for brand-new titles.",
        body,
    ))
    story.append(Spacer(1, 20))

    # --- Methodology ---
    story.append(Paragraph("Methodology", h2))
    steps = [
        "1. Clean the movie catalog and combine each movie's genres (and tags) into one text field.",
        "2. Vectorize that text with TF-IDF (term frequency-inverse document frequency), "
        "turning each movie into a numeric vector where rarer, more distinctive genres/tags "
        "carry more weight.",
        "3. Compute pairwise Cosine Similarity across all movie vectors — a score from 0 "
        "(nothing in common) to 1 (identical content profile).",
        "4. For a given movie, rank every other movie by similarity score and return the top N.",
    ]
    for s in steps:
        story.append(Paragraph(s, body))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 10))

    # --- Genre distribution chart ---
    story.append(Paragraph("Catalog Overview", h2))
    story.append(Image("genre_distribution.png", width=6.2 * inch, height=3.5 * inch))
    story.append(PageBreak())

    # --- Sample recommendations ---
    story.append(Paragraph("Example Recommendations", h2))
    story.append(Paragraph(
        "The following examples show recommendations generated for a few sample movies "
        "from the catalog, to illustrate how the system behaves.",
        body,
    ))
    story.append(Spacer(1, 10))

    for i, (movie_title, rec_df, chart_path) in enumerate(sample_results):
        story.append(Paragraph(f"Because you watched: \u201c{movie_title}\u201d", h2))
        story.append(Image(chart_path, width=6.2 * inch, height=3.5 * inch))
        story.append(Spacer(1, 8))

        table_data = [["#", "Recommended Movie", "Genres", "Similarity"]]
        for j, row in rec_df.head(5).iterrows():
            table_data.append([str(j + 1), row["title"], row["genres"], f"{row['similarity']:.3f}"])

        t = Table(table_data, hAlign="LEFT", colWidths=[0.3 * inch, 2.5 * inch, 2.4 * inch, 0.8 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5E35B1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDE7F6")]),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        if i < len(sample_results) - 1:
            story.append(PageBreak())

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Note: This is a content-based recommender using genres/tags only — it has no "
        "knowledge of user ratings or viewing behavior, so it will always recommend movies "
        "with similar content regardless of popularity. Combining this with a collaborative-"
        "filtering model (based on user ratings) is the natural next step.",
        small,
    ))

    doc.build(story)
    print(f"\nSaved PDF report -> {REPORT_PATH}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    movies, tags = load_data()
    df = clean_and_build_content(movies, tags)
    sim_matrix, _ = build_similarity_matrix(df)

    plot_genre_distribution(df, "genre_distribution.png")

    sample_titles = SAMPLE_MOVIES if SAMPLE_MOVIES else pick_sample_movies(df, n=3)

    sample_results = []
    for i, title in enumerate(sample_titles):
        rec_df = recommend(title, df, sim_matrix, top_n=TOP_N)
        print(f"\nTop {TOP_N} recommendations for '{title}':")
        print(rec_df)
        chart_path = f"recommendation_{i+1}.png"
        plot_recommendation_bar(title, rec_df, chart_path)
        sample_results.append((title, rec_df, chart_path))

    build_pdf_report(df, sample_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
