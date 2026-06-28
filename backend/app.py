from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator

from src.api.serialization import format_keyframe_id as format_keyframe_id_from_dict
from src.index.retrieval_backend import create_retrieval_backend
from src.ui import (
    split_query,
    rerank_multi_query,
    get_surrounding_frames,
    get_timestamp_from_row,
)

st.set_page_config(
    page_title="Frame Retrieval",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# CSS - Gemini-like UI
# =========================
st.markdown(
    """
<style>
:root {
    --bg: #0b0d12;
    --panel: #17191f;
    --panel-2: #202124;
    --text: #e8eaed;
    --muted: #9aa0a6;
    --blue: #4285f4;
    --border: rgba(255,255,255,0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 50% 35%, rgba(40,70,180,0.28), transparent 36%),
        linear-gradient(180deg, #0b0d12 0%, #090a0d 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2.2rem;
    max-width: 1500px;
}

.gemini-shell {
    min-height: 42vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    animation: fadeUp 0.8s ease-out;
}

.gemini-logo {
    font-size: 1.9rem;
    margin-bottom: 1.2rem;
    filter: drop-shadow(0 0 18px rgba(66,133,244,.65));
}

.gemini-title {
    font-size: clamp(2rem, 4vw, 3.1rem);
    font-weight: 500;
    letter-spacing: -0.04em;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #e8eaed, #a8c7fa, #d7aefb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.search-wrap {
    width: min(760px, 88vw);
    background: #1f1f1f;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.45rem 0.8rem;
    box-shadow: 0 24px 80px rgba(0,0,0,.35);
    transition: all .25s ease;
}

.search-wrap:hover {
    border-color: rgba(66,133,244,.45);
    box-shadow: 0 28px 90px rgba(66,133,244,.16);
}

div[data-testid="stTextInput"] input {
    background: transparent !important;
    color: var(--text) !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1.05rem;
}

div[data-testid="stTextInput"] label {
    display: none;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid var(--border);
    background: #202124;
    color: var(--text);
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: var(--blue);
    transform: translateY(-1px);
    box-shadow: 0 10px 30px rgba(66,133,244,.20);
}

.result-card {
    background: rgba(24,26,32,.88);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 0.7rem;
    margin-bottom: 1rem;
    transition: all .25s ease;
    animation: fadeUp .45s ease both;
}

.result-card:hover {
    transform: translateY(-5px);
    border-color: rgba(66,133,244,.45);
    box-shadow: 0 20px 50px rgba(0,0,0,.35);
}

.result-title {
    font-weight: 650;
    font-size: .95rem;
    margin-top: .55rem;
}

.result-meta {
    color: var(--muted);
    font-size: .78rem;
    margin-bottom: .45rem;
}

.result-card img {
    border-radius: 16px;
}

.sidebar-box {
    position: fixed;
    left: 18px;
    top: 18px;
    bottom: 18px;
    width: 52px;
    border-radius: 26px;
    background: rgba(15,17,22,.55);
    border: 1px solid var(--border);
    backdrop-filter: blur(18px);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 14px;
    gap: 18px;
    z-index: 999;
}

.side-icon {
    font-size: 1.15rem;
    opacity: .92;
}

.side-spacer {
    flex: 1;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stSidebar"] {
    background: #101114;
}
</style>

<div class="sidebar-box">
    <div class="side-icon">✦</div>
    <div class="side-icon">🔎</div>
    <div class="side-icon">🖼️</div>
    <div class="side-icon">🎬</div>
    <div class="side-spacer"></div>
    <div class="side-icon">⚙️</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# Cache
# =========================
@st.cache_data
def load_config_cached(path: str) -> dict:
    from src.config.retrieval_config import load_retrieval_config, to_dict

    cfg = load_retrieval_config(path)
    return to_dict(cfg)


@st.cache_resource
def load_retrieval_system(cfg: dict):
    return create_retrieval_backend(cfg)


# =========================
# Utils
# =========================
def translate_query(query: str, cfg: dict) -> str:
    source = cfg.get("translate", {}).get("source", "vi")
    target = cfg.get("translate", {}).get("target", "en")
    return GoogleTranslator(source=source, target=target).translate(query)


def format_keyframe_id(row: pd.Series) -> str:
    return format_keyframe_id_from_dict(row.to_dict())


def short_label(row: pd.Series) -> str:
    video_id = str(row.get("video_id", "unknown"))
    frame_id = format_keyframe_id(row)
    return f"{video_id}/{frame_id}"


def get_video_path(row: pd.Series, cfg: dict) -> Path | None:
    if "video_path" in row and isinstance(row["video_path"], str):
        path = Path(row["video_path"])
        if path.exists():
            return path

    videos_root = Path(cfg.get("paths", {}).get("videos_root", "data/videos"))
    dataset = str(row.get("dataset", ""))
    video_id = str(row.get("video_id", ""))

    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        for path in [
            videos_root / dataset / f"{video_id}{ext}",
            videos_root / f"{video_id}{ext}",
        ]:
            if path.exists():
                return path

    return None


def run_search(
    system, query: str, top_k: int, candidate_multiplier: int, use_split: bool
):
    sub_queries = split_query(query) if use_split else [query]
    candidate_k = top_k * candidate_multiplier

    all_results = []
    for sub_query in sub_queries:
        df = system.search(sub_query, top_k=candidate_k)
        if not df.empty:
            all_results.append(df)

    results = rerank_multi_query(all_results)

    if results.empty:
        return results, sub_queries

    if (
        "alignment_score" in results.columns
        and "retrieval_score" not in results.columns
    ):
        results["retrieval_score"] = results["alignment_score"]

    results = results.head(top_k).copy()
    results["display_rank"] = range(1, len(results) + 1)

    return results, sub_queries


# =========================
# Dialogs
# =========================
@st.dialog("🖼️ Surrounding Frames", width="large")
def surrounding_dialog(row_dict: dict, radius: int):
    row = pd.Series(row_dict)
    st.markdown(f"### `{short_label(row)}`")

    frames = get_surrounding_frames(row["keyframe_path"], radius=radius)

    if not frames:
        st.info("No surrounding frames found.")
        return

    cols = st.columns(min(len(frames), 8))

    for col, item in zip(cols, frames):
        with col:
            caption = item["keyframe_id"]
            if item.get("is_current"):
                caption = f"⭐ {caption}"

            st.image(
                str(item["path"]),
                caption=caption,
                use_container_width=True,
            )


@st.dialog("🎬 Play Video", width="large")
def play_dialog(row_dict: dict, cfg: dict):
    row = pd.Series(row_dict)
    st.markdown(f"### `{short_label(row)}`")

    timestamp = get_timestamp_from_row(row)
    video_path = get_video_path(row, cfg)

    if video_path is None:
        st.warning("Video file not found.")
        return

    st.caption(f"`{video_path.name}` — start `{int(timestamp or 0)}s`")
    st.video(str(video_path), start_time=int(timestamp or 0))


# =========================
# Render
# =========================
def render_card(row: pd.Series, rank: int, cfg: dict, radius: int):
    image_path = Path(row["keyframe_path"])

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning("Missing image")

    st.markdown(
        f"""
<div class="result-title">#{rank} · <code>{short_label(row)}</code></div>
<div class="result-meta">
cos={row.get("avg_score", 0):.3f} · 
score={row.get("retrieval_score", row.get("alignment_score", 0)):.3f}
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(
            "🖼️", key=f"sur_{rank}_{row['keyframe_path']}", help="Surrounding frames"
        ):
            surrounding_dialog(row.to_dict(), radius)

    with c2:
        if st.button("▶️", key=f"play_{rank}_{row['keyframe_path']}", help="Play video"):
            play_dialog(row.to_dict(), cfg)

    with c3:
        if st.button("❤️", key=f"like_{rank}_{row['keyframe_path']}", help="Like"):
            st.session_state.setdefault("liked", []).append(row.to_dict())
            st.toast("Added to selected frames")

    st.markdown("</div>", unsafe_allow_html=True)


def render_grid(results: pd.DataFrame, cfg: dict, radius: int, columns: int):
    for start in range(0, len(results), columns):
        cols = st.columns(columns)
        batch = results.iloc[start : start + columns]

        for col, (_, row) in zip(cols, batch.iterrows()):
            with col:
                render_card(
                    row=row,
                    rank=int(row["display_rank"]),
                    cfg=cfg,
                    radius=radius,
                )


# =========================
# Main
# =========================
def main():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        cfg_path = st.text_input("Config", "configs/app.yaml")
        cfg = load_config_cached(cfg_path)

        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=int(cfg["search"]["max_top_k"]),
            value=int(cfg["search"]["default_top_k"]),
        )

        grid_cols = st.slider("Grid columns", min_value=3, max_value=8, value=5)

        radius = st.slider(
            "Surrounding radius",
            min_value=1,
            max_value=int(cfg["ui"]["max_surrounding_radius"]),
            value=int(cfg["ui"]["surrounding_radius"]),
        )

        use_split = st.toggle("🧩 Split query", value=True)

        use_translate = st.toggle(
            "🌐 Translate VI → EN",
            value=bool(cfg.get("translate", {}).get("enabled_default", False)),
        )

    system = load_retrieval_system(cfg)

    st.markdown('<section class="gemini-shell">', unsafe_allow_html=True)
    st.markdown('<div class="gemini-logo">✦</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gemini-title">Hi Tân, what frame are you looking for?</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)

    with st.form("search_form", clear_on_submit=False):
        q_col, b_col = st.columns([8, 1])

        with q_col:
            query = st.text_input(
                "Query",
                placeholder="Ask for an event, action, object, scene...",
                label_visibility="collapsed",
            )

        with b_col:
            submitted = st.form_submit_button("🔎")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</section>", unsafe_allow_html=True)

    if submitted:
        query = query.strip()

        if not query:
            st.warning("Please enter a query.")
            return

        original_query = query

        if use_translate:
            with st.spinner("Translating query..."):
                query = translate_query(query, cfg)

        with st.spinner("Retrieving relevant frames..."):
            results, sub_queries = run_search(
                system=system,
                query=query,
                top_k=top_k,
                candidate_multiplier=int(cfg["search"]["candidate_multiplier"]),
                use_split=use_split,
            )

        st.session_state["results"] = results
        st.session_state["sub_queries"] = sub_queries
        st.session_state["query"] = query
        st.session_state["original_query"] = original_query

    if "query" not in st.session_state:
        return

    q = st.session_state["query"]
    oq = st.session_state.get("original_query", q)

    if oq != q:
        st.caption(f"Original: `{oq}`")
        st.caption(f"Translated: `{q}`")
    else:
        st.caption(f"Query: `{q}`")

    with st.expander("🧩 Sub-queries", expanded=False):
        st.write(st.session_state.get("sub_queries", []))

    results = st.session_state.get("results")

    if results is None:
        return

    if results.empty:
        st.warning("No results found.")
        return

    topbar = st.columns([7, 1])

    with topbar[0]:
        st.markdown("## Retrieved Frames")

    with topbar[1]:
        if st.button("📤 Submit"):
            st.success("Submitted selected results.")

    render_grid(
        results=results,
        cfg=cfg,
        radius=radius,
        columns=grid_cols,
    )


if __name__ == "__main__":
    main()
