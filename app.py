"""
Tennis Prediction Model — Streamlit App

Pages
-----
  1. Daily Slate     — today's predictions with edge highlights and filters
  2. Match Deep Dive — ad-hoc prediction for any two players
  3. Model Performance — Brier score, CLV, accuracy by tier / surface
  4. Settings        — API key, thresholds, tour toggles

Run
---
  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── must be first streamlit call ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "Tennis Predictions",
    page_icon   = "🎾",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

ROOT            = Path(__file__).parent
PREDICTIONS_CSV = ROOT / "output" / "predictions.csv"
CLV_CSV         = ROOT / "output" / "clv_tracker.csv"
CONFIG_JSON     = ROOT / "config.json"
ENV_FILE        = ROOT / ".env"

sys.path.insert(0, str(ROOT))

# ── global CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Tighten sidebar */
[data-testid="stSidebar"] { min-width: 200px; max-width: 220px; }
/* Badge helpers */
.badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
.badge-atp  { background:#1565c0; color:#fff; }
.badge-wta  { background:#880e4f; color:#fff; }
.badge-ch   { background:#424242; color:#fff; }
.badge-hard { background:#1a3a5c; color:#9ecbff; }
.badge-clay { background:#4a2e1a; color:#ffb74d; }
.badge-grass{ background:#1b3d1f; color:#81c784; }
.badge-sharp   { background:#00695c; color:#e0f2f1; }
.badge-moderate{ background:#e65100; color:#fff3e0; }
.badge-wide    { background:#4a148c; color:#f3e5f5; }
/* Row highlights painted via dataframe styling */
/* Mobile: stack columns */
@media (max-width: 768px) {
  [data-testid="column"] { min-width: 100% !important; }
}
/* Hide Streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── config helpers ─────────────────────────────────────────────────────────────

_DEFAULT_CFG: dict[str, Any] = {
    "odds_api_key":    "",
    "edge_threshold":  4,
    "min_tier":        "WIDE",
    "tours":           ["ATP", "WTA", "Challenger"],
    "surfaces":        ["hard", "clay", "grass"],
}


def load_config() -> dict:
    if CONFIG_JSON.exists():
        try:
            with open(CONFIG_JSON) as f:
                return {**_DEFAULT_CFG, **json.load(f)}
        except Exception:
            pass
    return dict(_DEFAULT_CFG)


def save_config(cfg: dict) -> None:
    with open(CONFIG_JSON, "w") as f:
        json.dump(cfg, f, indent=2)


def save_env_key(key: str, value: str) -> None:
    """Upsert KEY=value in .env file."""
    lines: list[str] = []
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            lines = f.readlines()
    prefix = f"{key}="
    new_line = f"{key}={value}\n"
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)


def effective_api_key(cfg: dict) -> str:
    """Return API key. Priority: st.secrets → env var → config.json."""
    try:
        key = st.secrets.get("ODDS_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ODDS_API_KEY") or cfg.get("odds_api_key") or ""


# ── cloud / local environment detection ───────────────────────────────────────

def _model_files_present() -> bool:
    """True when the three pkl files needed for inference are on disk."""
    saved = ROOT / "models" / "saved"
    return all((saved / f).exists() for f in
               ("xgb_calibrated.pkl", "feature_list.json"))


def _data_present() -> bool:
    """True when processed data (needed by PipelineContext) is on disk."""
    return (ROOT / "data" / "processed").exists() and any(
        (ROOT / "data" / "processed").iterdir()
    )


# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_CSV)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["prob_a", "prob_b", "model_edge", "prob_low", "prob_high",
                "confidence_width", "opening_odds_a", "opening_odds_b"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Infer tour / confidence_tier if missing (older schema)
    if "tour" not in df.columns:
        df["tour"] = "atp"
    if "confidence_tier" not in df.columns:
        df["confidence_tier"] = None
    return df


@st.cache_data(ttl=60)
def load_clv() -> pd.DataFrame:
    if not CLV_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(CLV_CSV)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["model_prob_a", "opening_implied_a", "closing_implied_a",
                "clv_delta", "result", "sharp_flag", "movement_magnitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── model / context (cached across sessions) ──────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def _load_model_only():
    """Load ensemble model + CI. Does NOT need the processed data directory."""
    from predict import load_model
    return load_model()  # (model, feat_cols, meta, ci)


@st.cache_resource(show_spinner="Loading player database…")
def _load_ctx():
    """Load PipelineContext (needs data/processed/). Returns None if unavailable."""
    try:
        from predict import build_name_lookup
        from features.pipeline import PipelineContext
        ctx = PipelineContext.load()
        norm_lookup = build_name_lookup(ctx.name_to_id)
        return ctx, norm_lookup
    except Exception:
        return None, {}


# ── predict helpers ───────────────────────────────────────────────────────────

def run_predict_subprocess(api_key: str) -> tuple[bool, str]:
    """
    Call predict.py in a subprocess so Streamlit doesn't block the event loop.
    Returns (success, output_text).
    """
    env = {**os.environ, "ODDS_API_KEY": api_key}
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "predict.py"), "--dry-run"],
            capture_output=True, text=True, timeout=300, env=env,
        )
        out = (result.stdout + result.stderr).strip()
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Prediction timed out after 5 minutes."
    except Exception as exc:
        return False, str(exc)


def run_predict_live(api_key: str) -> tuple[bool, str]:
    """Run predict.py (writes to predictions.csv)."""
    env = {**os.environ, "ODDS_API_KEY": api_key}
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "predict.py")],
            capture_output=True, text=True, timeout=300, env=env,
        )
        out = (result.stdout + result.stderr).strip()
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Prediction timed out."
    except Exception as exc:
        return False, str(exc)


# ── display helpers ───────────────────────────────────────────────────────────

_TIER_BADGE = {
    "SHARP":    '<span class="badge badge-sharp">SHARP</span>',
    "MODERATE": '<span class="badge badge-moderate">MOD</span>',
    "WIDE":     '<span class="badge badge-wide">WIDE</span>',
}
_SURF_BADGE = {
    "hard":  '<span class="badge badge-hard">HARD</span>',
    "clay":  '<span class="badge badge-clay">CLAY</span>',
    "grass": '<span class="badge badge-grass">GRASS</span>',
}
_TOUR_BADGE = {
    "atp":        '<span class="badge badge-atp">ATP</span>',
    "wta":        '<span class="badge badge-wta">WTA</span>',
    "challenger": '<span class="badge badge-ch">CH</span>',
}


def _edge_color(edge: float, threshold: float) -> str:
    if edge > threshold / 100:
        return "color:#00d96e; font-weight:700"
    if edge < -threshold / 100:
        return "color:#f85149; font-weight:700"
    return ""


def _style_slate_df(df: pd.DataFrame, threshold_pct: float) -> pd.io.formats.style.Styler:
    """Apply green / red row highlighting based on model_edge."""
    GREEN_BG = "background-color:#0d2818; color:#e6edf3"
    RED_BG   = "background-color:#2d0f0f; color:#e6edf3"

    def row_style(row):
        edge = row.get("Edge %", None)
        if pd.isna(edge):
            return [""] * len(row)
        try:
            val = float(str(edge).replace("%", "").replace("+", ""))
        except ValueError:
            return [""] * len(row)
        if val > threshold_pct:
            return [GREEN_BG] * len(row)
        if val < -threshold_pct:
            return [RED_BG] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


def _format_slate_display(group: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Format a group (one tournament) into display columns."""
    rows = []
    for _, r in group.iterrows():
        prob_a = r.get("prob_a", np.nan)
        prob_b = 1.0 - prob_a if pd.notna(prob_a) else np.nan
        edge   = r.get("model_edge", np.nan)
        pl     = r.get("prob_low", np.nan)
        ph     = r.get("prob_high", np.nan)
        tier   = r.get("confidence_tier", "")
        sharp  = r.get("sharp_flag", np.nan)

        ci_str = (
            f"{pl:.0%} – {ph:.0%}"
            if pd.notna(pl) and pd.notna(ph) else "—"
        )
        edge_str = f"{edge:+.1%}" if pd.notna(edge) else "—"
        sharp_str = "⚡" if pd.notna(sharp) and float(sharp) == 1.0 else ""

        rows.append({
            "Player A":    r.get("player_a", ""),
            "Model A %":   f"{prob_a:.1%}" if pd.notna(prob_a) else "—",
            "Player B":    r.get("player_b", ""),
            "Model B %":   f"{prob_b:.1%}" if pd.notna(prob_b) else "—",
            "Edge %":      edge_str,
            "CI":          ci_str,
            "Tier":        str(tier) if tier else "—",
            "⚡":          sharp_str,
            "_edge_raw":   float(edge) * 100 if pd.notna(edge) else 0.0,
            "_idx":        r.name,
        })

    return pd.DataFrame(rows)


# ── Page 1: Daily Slate ────────────────────────────────────────────────────────

def page_daily_slate(cfg: dict) -> None:
    # Auto-refresh every 30 min
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1_800_000, limit=None, key="slate_autorefresh")
    except ImportError:
        pass

    api_key = effective_api_key(cfg)
    edge_threshold = float(cfg.get("edge_threshold", 4))

    # ── header ──────────────────────────────────────────────────────────────
    hcol, rcol = st.columns([5, 1])
    with hcol:
        st.markdown(
            f"## 🎾 Daily Slate &nbsp;·&nbsp; "
            f"<span style='color:#8b949e'>{date.today().strftime('%B %d, %Y')}</span>",
            unsafe_allow_html=True,
        )
    with rcol:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            if not api_key:
                st.error("Set your Odds API key in Settings first.")
            else:
                with st.spinner("Fetching odds and running predictions…"):
                    ok, msg = run_predict_live(api_key)
                load_predictions.clear()
                load_clv.clear()
                if ok:
                    st.success("Predictions updated.")
                else:
                    st.warning(f"Completed with warnings:\n```\n{msg[-600:]}\n```")
                st.rerun()

    # ── API key warning ──────────────────────────────────────────────────────
    if not api_key:
        st.warning(
            "⚠️ **No Odds API key configured.** Showing cached predictions only.  "
            "Add your key in **Settings** to fetch live odds.",
            icon="⚠️",
        )

    # ── load data ────────────────────────────────────────────────────────────
    df = load_predictions()

    if df.empty:
        st.info(
            "No predictions yet. Click **🔄 Refresh** to fetch today's slate, "
            "or run `python predict.py` from the terminal.",
        )
        return

    st.caption(
        f"Loaded {len(df):,} predictions · "
        f"last modified {datetime.fromtimestamp(PREDICTIONS_CSV.stat().st_mtime).strftime('%H:%M')}"
    )

    # ── filters ──────────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        with fc1:
            min_edge = st.slider(
                "Min |edge| %", -30, 30, 0, step=1,
                help="Show only matches where model edge exceeds this value",
            )
        with fc2:
            avail_tiers = [t for t in ["SHARP", "MODERATE", "WIDE"]
                           if t in df["confidence_tier"].values]
            tier_filter = st.multiselect(
                "Confidence tier",
                options=["SHARP", "MODERATE", "WIDE"],
                default=avail_tiers if avail_tiers else ["SHARP", "MODERATE", "WIDE"],
            )
        with fc3:
            tours_in_data = [t.upper() for t in df["tour"].dropna().unique()]
            tour_opts = sorted(set(["ATP", "WTA", "Challenger"]) | set(tours_in_data))
            tour_filter = st.multiselect("Tour", tour_opts, default=tour_opts)
        with fc4:
            surfs_in_data = [s.lower() for s in df["surface"].dropna().unique()]
            surf_opts = sorted(set(["hard", "clay", "grass"]) | set(surfs_in_data))
            surf_filter = st.multiselect("Surface", surf_opts, default=surf_opts)

    # Apply filters
    fdf = df.copy()
    if "model_edge" in fdf.columns:
        fdf = fdf[fdf["model_edge"].isna() | (fdf["model_edge"] * 100 >= min_edge)]
    if tier_filter and "confidence_tier" in fdf.columns:
        fdf = fdf[fdf["confidence_tier"].isin(tier_filter) | fdf["confidence_tier"].isna()]
    if tour_filter:
        fdf = fdf[fdf["tour"].str.upper().isin([t.upper() for t in tour_filter])]
    if surf_filter:
        fdf = fdf[fdf["surface"].str.lower().isin([s.lower() for s in surf_filter])]

    if fdf.empty:
        st.info("No matches match the current filters.")
        return

    # Summary stats bar
    n_sharp = (fdf["confidence_tier"] == "SHARP").sum() if "confidence_tier" in fdf.columns else 0
    n_edge  = ((fdf["model_edge"].fillna(0) * 100) > edge_threshold).sum()
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Matches",     len(fdf))
    mc2.metric("SHARP",       n_sharp)
    mc3.metric(f"Edge >{edge_threshold:.0f}%", n_edge)
    mc4.metric("Tournaments", fdf["tournament"].nunique() if "tournament" in fdf.columns else 0)
    st.divider()

    # ── per-tournament tables ─────────────────────────────────────────────────
    tourn_col = "tournament" if "tournament" in fdf.columns else None
    groups = (
        fdf.groupby(tourn_col, sort=False)
        if tourn_col else [(None, fdf)]
    )

    for tournament, group in groups:
        group = group.reset_index(drop=True)
        first = group.iloc[0]
        surface   = str(first.get("surface", "?")).lower()
        tour_val  = str(first.get("tour",    "atp")).lower()
        surf_html = _SURF_BADGE.get(surface,  f'<span class="badge">{surface.upper()}</span>')
        tour_html = _TOUR_BADGE.get(tour_val, f'<span class="badge">{tour_val.upper()}</span>')

        label = tournament or "Unknown Tournament"
        st.markdown(
            f"<h4 style='margin-bottom:4px'>{label}&nbsp;&nbsp;{surf_html}&nbsp;{tour_html}</h4>",
            unsafe_allow_html=True,
        )

        display = _format_slate_display(group, edge_threshold)
        visible = display.drop(columns=["_edge_raw", "_idx"])

        styled = _style_slate_df(visible, edge_threshold)

        sel_key = f"sel_{label.replace(' ','_')}"
        event = st.dataframe(
            styled,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key=sel_key,
            hide_index=True,
            column_config={
                "Tier": st.column_config.TextColumn(width="small"),
                "⚡":   st.column_config.TextColumn(width="small", label="⚡"),
                "CI":   st.column_config.TextColumn(width="medium"),
            },
        )

        if event.selection.rows:
            sel_i   = event.selection.rows[0]
            src_idx = int(display.iloc[sel_i]["_idx"])
            src_row = df.loc[src_idx]
            with st.container(border=True):
                _show_match_detail_inline(src_row)

        st.markdown("")   # spacer


def _show_match_detail_inline(row: pd.Series) -> None:
    """Inline expanded detail for a selected match row."""
    prob_a = float(row.get("prob_a", 0.5))
    prob_b = 1.0 - prob_a
    pl  = row.get("prob_low",  None)
    ph  = row.get("prob_high", None)
    tier = row.get("confidence_tier", "")

    st.markdown(
        f"**{row.get('player_a', '?')}** vs **{row.get('player_b', '?')}** "
        f"· {row.get('tournament','?')} · {str(row.get('surface','?')).upper()}"
    )

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Model (A wins)", f"{prob_a:.1%}")
    if pd.notna(pl) and pd.notna(ph):
        pc2.metric("CI", f"{pl:.1%} – {ph:.1%}")
    if pd.notna(row.get("model_edge")):
        pc3.metric("Edge", f"{row['model_edge']:+.1%}")

    # Probability gauge bar
    fig = go.Figure(go.Bar(
        x=[prob_a * 100, prob_b * 100],
        y=[row.get("player_a", "A"), row.get("player_b", "B")],
        orientation="h",
        marker_color=["#00d96e", "#f85149"],
        text=[f"{prob_a:.1%}", f"{prob_b:.1%}"],
        textposition="inside",
    ))
    fig.update_layout(
        height=120, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, xaxis=dict(range=[0,100], showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
        font=dict(color="#e6edf3"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key drivers
    drivers = row.get("key_drivers", "")
    if drivers and str(drivers) not in ("", "nan", "balanced"):
        st.markdown("**Drivers:** " + str(drivers))

    # Odds
    oa = row.get("opening_odds_a")
    ob = row.get("opening_odds_b")
    if pd.notna(oa) and pd.notna(ob):
        st.caption(f"Opening odds:  {row.get('player_a','A')} {float(oa):.2f}  ·  "
                   f"{row.get('player_b','B')} {float(ob):.2f}")


# ── Page 2: Match Deep Dive ───────────────────────────────────────────────────

def page_deep_dive(_cfg: dict) -> None:
    st.markdown("## 🔍 Match Deep Dive")
    st.caption("Type any two player names and get a full prediction with feature breakdown.")

    with st.form("deep_dive_form"):
        c1, c2 = st.columns(2)
        player_a = c1.text_input("Player A (favourite / higher ranked)", placeholder="e.g. Carlos Alcaraz")
        player_b = c2.text_input("Player B", placeholder="e.g. Novak Djokovic")

        d1, d2, d3, d4 = st.columns(4)
        surface    = d1.selectbox("Surface",    ["hard", "clay", "grass"])
        tour       = d2.selectbox("Tour",       ["atp", "wta"])
        best_of    = d3.selectbox("Best of",    [3, 5])
        tournament = d4.text_input("Tournament", value="Unknown")

        date_inp = st.date_input("Match date", value=date.today())

        c_oa, c_ob = st.columns(2)
        odds_a = c_oa.number_input("Odds Player A (decimal)", min_value=1.0, value=2.0, step=0.05)
        odds_b = c_ob.number_input("Odds Player B (decimal)", min_value=1.0, value=2.0, step=0.05)

        submitted = st.form_submit_button("🎯 Predict", use_container_width=True)

    if not submitted:
        st.info("Enter two player names above and click **🎯 Predict**.")
        return

    if not player_a or not player_b:
        st.error("Both player names are required.")
        return

    with st.spinner("Loading model and computing prediction…"):
        try:
            model, feat_cols, _meta, ci = _load_model_only()
        except Exception as exc:
            st.error(f"Could not load model: {exc}")
            return

        # PipelineContext requires processed data — not available on Streamlit Cloud
        ctx, norm_lookup = _load_ctx()

        if ctx is None:
            st.warning(
                "Player database not available in this deployment — the processed match data "
                "(`data/processed/`) is not included in the repo due to size (283 MB).  \n"
                "**Match Deep Dive is fully functional when running locally** with the complete "
                "dataset. On Streamlit Cloud, use the **Daily Slate** page to view predictions "
                "generated from your local machine and pushed to the repo.",
                icon="ℹ️",
            )
            return

        from predict import resolve_player_id, predict_match

        a_id = resolve_player_id(player_a, ctx.name_to_id, norm_lookup)
        b_id = resolve_player_id(player_b, ctx.name_to_id, norm_lookup)

        if a_id is None:
            st.error(f"Could not find player: **{player_a}**. Check spelling or try initials.")
            return
        if b_id is None:
            st.error(f"Could not find player: **{player_b}**. Check spelling or try initials.")
            return

        try:
            result = predict_match(
                ctx, model, feat_cols,
                player_a_id   = a_id,
                player_b_id   = b_id,
                player_a_name = player_a,
                player_b_name = player_b,
                surface       = surface,
                tournament    = tournament,
                match_date    = pd.Timestamp(date_inp),
                best_of       = best_of,
                tour          = tour,
                opening_odds_a = float(odds_a) if odds_a > 1 else None,
                opening_odds_b = float(odds_b) if odds_b > 1 else None,
                ci             = ci,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    # ── Results ──────────────────────────────────────────────────────────────
    prob_a = result["prob_a"]
    prob_b = result["prob_b"]
    pl = result.get("prob_low")
    ph = result.get("prob_high")
    tier = result.get("confidence_tier", "")
    edge = result.get("model_edge")

    st.divider()
    st.markdown(f"### {player_a}  vs  {player_b}")
    st.caption(f"{tournament} · {surface.upper()} · {tour.upper()} · best of {best_of}")

    # Win probability bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=player_a,
        x=[prob_a * 100], y=[""], orientation="h",
        marker_color="#00d96e",
        text=f"{prob_a:.1%}", textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name=player_b,
        x=[prob_b * 100], y=[""], orientation="h",
        marker_color="#f85149",
        text=f"{prob_b:.1%}", textposition="inside",
    ))
    fig.update_layout(
        barmode="stack", height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1,
                    font=dict(color="#e6edf3")),
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
        font=dict(color="#e6edf3"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{player_a} wins", f"{prob_a:.1%}")
    m2.metric("CI", f"{pl:.1%} – {ph:.1%}" if pd.notna(pl) and pd.notna(ph) else "—")
    if tier:
        m3.metric("Confidence", tier)
    if edge is not None and pd.notna(edge):
        m4.metric("Model edge", f"{edge:+.1%}")

    st.divider()

    # ── Top feature drivers bar chart ─────────────────────────────────────────
    feat_row = result.get("feat_row", {})
    if feat_row:
        _show_feature_chart(feat_row, player_a, player_b)

    # ── H2H summary ──────────────────────────────────────────────────────────
    h2h_n   = feat_row.get("h2h_n_overall", np.nan)
    h2h_pct = feat_row.get("h2h_win_pct_surface", feat_row.get("h2h_win_pct_overall", np.nan))
    if pd.notna(h2h_n) and float(h2h_n) >= 1:
        st.markdown("**H2H (surface-specific):**")
        wins_a = round(float(h2h_n) * float(h2h_pct)) if pd.notna(h2h_pct) else "?"
        st.markdown(
            f"  {player_a} **{wins_a}** – **{round(float(h2h_n) - float(wins_a)) if wins_a != '?' else '?'}** {player_b} "
            f"over {int(h2h_n)} matches"
        )
    else:
        st.caption("No H2H data (< 1 recorded match on this surface).")

    # ── Raw feature table (collapsed) ────────────────────────────────────────
    with st.expander("📋 All feature values"):
        feat_df = (
            pd.DataFrame([{"Feature": k, "Value": v}
                          for k, v in feat_row.items()
                          if k != "label" and pd.notna(v)])
            .reset_index(drop=True)
        )
        feat_df["Value"] = feat_df["Value"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


def _show_feature_chart(feat_row: dict, player_a: str, player_b: str) -> None:
    """Top-10 feature drivers bar chart."""
    _DRIVER_FEATURES = [
        ("elo_diff",               "Elo advantage"),
        ("a_hold_pct",             f"{player_a} hold %"),
        ("b_hold_pct",             f"{player_b} hold %"),
        ("a_form_weighted_win_pct",f"{player_a} form"),
        ("b_form_weighted_win_pct",f"{player_b} form"),
        ("h2h_win_pct_surface",    "H2H (surface)"),
        ("a_first_srv_won_pct",    f"{player_a} 1st srv won"),
        ("b_first_srv_won_pct",    f"{player_b} 1st srv won"),
        ("a_break_pct",            f"{player_a} break %"),
        ("b_break_pct",            f"{player_b} break %"),
        ("a_form_top10_win_pct",   f"{player_a} vs Top 10"),
        ("court_speed_index",      "Court speed"),
        ("a_injury_flag",          f"{player_a} injury flag"),
        ("b_injury_flag",          f"{player_b} injury flag"),
        ("rally_0_4_delta",        "Rally ≤4 shots edge"),
        ("rally_9plus_delta",      "Rally 9+ shots edge"),
    ]

    rows = []
    for key, label in _DRIVER_FEATURES:
        val = feat_row.get(key, np.nan)
        if pd.notna(val):
            rows.append({"Feature": label, "Value": float(val)})

    if not rows:
        return

    driver_df = pd.DataFrame(rows).head(10)
    driver_df = driver_df.sort_values("Value", ascending=True)
    colors = ["#00d96e" if v >= 0 else "#f85149" for v in driver_df["Value"]]

    fig = px.bar(
        driver_df, x="Value", y="Feature", orientation="h",
        title="Top Feature Drivers",
        color="Value",
        color_continuous_scale=[[0, "#f85149"], [0.5, "#8b949e"], [1, "#00d96e"]],
        color_continuous_midpoint=0,
    )
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#30363d", zeroline=True, zerolinecolor="#8b949e"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page 3: Model Performance ─────────────────────────────────────────────────

def page_performance(_cfg: dict) -> None:
    st.markdown("## 📈 Model Performance")

    df = load_clv()

    if df.empty:
        st.info("No CLV data yet. Predictions appear here once `clv_tracker.csv` has rows.")
        return

    has_results = df["result"].notna().any() if "result" in df.columns else False

    # ── Summary metrics ──────────────────────────────────────────────────────
    n_pred = len(df)
    n_res  = int(df["result"].notna().sum()) if "result" in df.columns else 0
    n_clv  = int(df["clv_delta"].notna().sum()) if "clv_delta" in df.columns else 0

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total predictions", f"{n_pred:,}")
    sc2.metric("Results filled",    f"{n_res:,}")
    sc3.metric("CLV data points",   f"{n_clv:,}")
    st.divider()

    if not has_results:
        st.info(
            "Results haven't been filled in yet. "
            "Run `python -m utils.result_logger` to backfill results from Sackmann CSVs.",
        )
        # Still show CLV distribution if available
        if n_clv > 0:
            _show_clv_section(df)
        return

    # ── Brier score over time ─────────────────────────────────────────────────
    st.markdown("### Rolling Brier Score (30-match window)")
    res_df = df[df["result"].notna() & df["model_prob_a"].notna()].copy()
    res_df = res_df.sort_values("date").reset_index(drop=True)
    res_df["sq_err"] = (res_df["model_prob_a"] - res_df["result"]) ** 2
    res_df["rolling_brier"] = res_df["sq_err"].rolling(30, min_periods=5).mean()

    overall_brier = float(res_df["sq_err"].mean())
    st.caption(f"Overall Brier score: **{overall_brier:.5f}** (lower = better; random = 0.25)")

    if res_df["rolling_brier"].notna().any():
        fig = px.line(
            res_df.dropna(subset=["rolling_brier"]),
            x="date", y="rolling_brier",
            labels={"rolling_brier": "Brier (30-match rolling)", "date": ""},
            color_discrete_sequence=["#00d96e"],
        )
        fig.add_hline(y=0.25, line_dash="dash", line_color="#8b949e",
                      annotation_text="Random baseline (0.25)")
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            yaxis=dict(gridcolor="#30363d"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Accuracy by confidence threshold ─────────────────────────────────────
    st.markdown("### Accuracy at Confidence Thresholds")
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    acc_rows = []
    for t in thresholds:
        mask = res_df["model_prob_a"] >= t
        sub  = res_df[mask]
        if len(sub) >= 5:
            acc = (sub["model_prob_a"].round(0) == sub["result"]).mean()
            acc_rows.append({
                "Threshold": f"≥ {t:.0%}",
                "N":  len(sub),
                "Accuracy": acc,
            })
    if acc_rows:
        acc_df = pd.DataFrame(acc_rows)
        fig = px.bar(
            acc_df, x="Threshold", y="Accuracy",
            text=acc_df["Accuracy"].apply(lambda x: f"{x:.1%}"),
            color="Accuracy",
            color_continuous_scale=[[0, "#f85149"], [0.5, "#e3b341"], [1, "#00d96e"]],
            color_continuous_midpoint=0.6,
            labels={"Accuracy": "Accuracy"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="#8b949e",
                      annotation_text="50% baseline")
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            coloraxis_showscale=False,
            yaxis=dict(tickformat=".0%", gridcolor="#30363d"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Breakdown by surface ──────────────────────────────────────────────────
    if "surface" in res_df.columns:
        st.markdown("### Brier Score by Surface")
        surf_brier = (
            res_df.groupby("surface")["sq_err"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "Brier", "count": "N"})
            .reset_index()
        )
        fig = px.bar(
            surf_brier, x="surface", y="Brier",
            text=surf_brier["Brier"].apply(lambda x: f"{x:.4f}"),
            color="surface",
            color_discrete_map={"hard": "#1565c0", "clay": "#bf6600", "grass": "#2e7d32"},
        )
        fig.add_hline(y=overall_brier, line_dash="dot", line_color="#8b949e",
                      annotation_text="Overall")
        fig.update_layout(
            height=240, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            showlegend=False,
            yaxis=dict(gridcolor="#30363d"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Sharp flag accuracy ───────────────────────────────────────────────────
    if "sharp_flag" in res_df.columns and res_df["sharp_flag"].notna().any():
        st.divider()
        st.markdown("### Sharp Flag Accuracy")
        res_df["correct"] = (
            (res_df["model_prob_a"] >= 0.5) == (res_df["result"] == 1)
        ).astype(float)
        sharp_acc = (
            res_df.groupby("sharp_flag")["correct"]
            .agg(["mean", "count"])
            .reset_index()
        )
        sharp_acc["Label"] = sharp_acc["sharp_flag"].map({1.0: "⚡ Sharp", 0.0: "Normal"})
        st.dataframe(
            sharp_acc[["Label", "mean", "count"]]
            .rename(columns={"mean": "Accuracy", "count": "N"}),
            use_container_width=True, hide_index=True,
        )

    # ── CLV section ───────────────────────────────────────────────────────────
    _show_clv_section(df)


def _show_clv_section(df: pd.DataFrame) -> None:
    """CLV distribution histogram."""
    clv_df = df[df["clv_delta"].notna()].copy()
    if clv_df.empty:
        return

    st.divider()
    st.markdown("### CLV Distribution")
    st.caption(
        f"CLV = model probability − closing implied probability.  "
        f"Positive = beating the closing line.  "
        f"Mean: **{clv_df['clv_delta'].mean():+.3f}**"
    )

    fig = px.histogram(
        clv_df, x="clv_delta", nbins=40,
        labels={"clv_delta": "CLV delta"},
        color_discrete_sequence=["#00d96e"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#f85149")
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        yaxis=dict(gridcolor="#30363d"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page 4: Settings ──────────────────────────────────────────────────────────

def page_settings(cfg: dict) -> None:
    st.markdown("## ⚙️ Settings")

    with st.form("settings_form"):
        st.markdown("### 🔑 API")
        api_key = st.text_input(
            "Odds API Key",
            value=cfg.get("odds_api_key", ""),
            type="password",
            help="From the-odds-api.com — stored locally in .env",
        )
        if os.environ.get("ODDS_API_KEY"):
            st.caption("✅ ODDS_API_KEY is set in environment (overrides saved key).")

        st.markdown("### 📊 Display")
        edge_threshold = st.slider(
            "Edge highlight threshold (%)",
            min_value=1, max_value=20,
            value=int(cfg.get("edge_threshold", 4)),
            help="Rows with |edge| above this are highlighted green/red",
        )
        min_tier = st.selectbox(
            "Minimum confidence tier to show",
            options=["WIDE", "MODERATE", "SHARP"],
            index=["WIDE", "MODERATE", "SHARP"].index(cfg.get("min_tier", "WIDE")),
        )

        st.markdown("### 🎾 Tours")
        tc1, tc2, tc3 = st.columns(3)
        atp_on         = tc1.checkbox("ATP Tour",   value="ATP" in cfg.get("tours", ["ATP", "WTA", "Challenger"]))
        wta_on         = tc2.checkbox("WTA Tour",   value="WTA" in cfg.get("tours", ["ATP", "WTA", "Challenger"]))
        challenger_on  = tc3.checkbox("Challenger", value="Challenger" in cfg.get("tours", ["ATP", "WTA", "Challenger"]))

        st.markdown("### 🏟️ Surfaces")
        sc1, sc2, sc3 = st.columns(3)
        hard_on  = sc1.checkbox("Hard",  value="hard"  in cfg.get("surfaces", ["hard", "clay", "grass"]))
        clay_on  = sc2.checkbox("Clay",  value="clay"  in cfg.get("surfaces", ["hard", "clay", "grass"]))
        grass_on = sc3.checkbox("Grass", value="grass" in cfg.get("surfaces", ["hard", "clay", "grass"]))

        submitted = st.form_submit_button("💾 Save Settings", use_container_width=True)

    if submitted:
        tours    = [t for t, on in [("ATP", atp_on), ("WTA", wta_on), ("Challenger", challenger_on)] if on]
        surfaces = [s for s, on in [("hard", hard_on), ("clay", clay_on), ("grass", grass_on)] if on]

        new_cfg = {
            **cfg,
            "odds_api_key":  api_key,
            "edge_threshold": edge_threshold,
            "min_tier":      min_tier,
            "tours":         tours,
            "surfaces":      surfaces,
        }
        save_config(new_cfg)

        if api_key:
            save_env_key("ODDS_API_KEY", api_key)
            os.environ["ODDS_API_KEY"] = api_key

        st.success("✅ Settings saved.")
        st.rerun()

    st.divider()
    st.markdown("### 🗂️ Data paths")
    st.code(
        f"Predictions : {PREDICTIONS_CSV}\n"
        f"CLV tracker : {CLV_CSV}\n"
        f"Config      : {CONFIG_JSON}\n"
        f"Env file    : {ENV_FILE}\n"
        f"Models dir  : {ROOT / 'models' / 'saved'}"
    )

    st.markdown("### 🧹 Cache")
    if st.button("Clear data cache (forces reload of CSVs)"):
        load_predictions.clear()
        load_clv.clear()
        st.success("Cache cleared.")


# ── Page 0: Setup instructions (shown when model files are absent) ─────────────

def page_setup() -> None:
    st.markdown("## Setup Required")
    st.error(
        "Model files not found. The trained model pkl files must be present in "
        "`models/saved/` before the app can make predictions.",
        icon="🚨",
    )

    st.markdown("### Quick-start instructions")
    st.markdown("""
1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/yashmadan2018/tennis-prediction-model.git
   cd tennis-prediction-model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download match data** (Jeff Sackmann CSVs → `data/raw/`):
   ```bash
   git clone https://github.com/JeffSackmann/tennis_atp.git data/raw/tennis_atp
   git clone https://github.com/JeffSackmann/tennis_wta.git data/raw/tennis_wta
   ```

4. **Process data and build features:**
   ```bash
   python -m utils.data_loader
   ```

5. **Train the model:**
   ```bash
   python models/train.py
   python models/ensemble.py
   ```

6. **Run predictions:**
   ```bash
   python predict.py
   ```

Once `models/saved/xgb_calibrated.pkl` exists, restart this app and all pages will be available.
    """)

    st.divider()
    st.markdown("### Streamlit Cloud deployment")
    st.markdown("""
The model pkl files (≈4 MB) are committed to the GitHub repo and will be present on Streamlit Cloud automatically.
The large data files (`data/raw/`, `data/processed/`) are **not** included — they are only needed for local re-training.

To deploy on Streamlit Cloud:
1. Push the repo to GitHub (model pkl files included).
2. Connect at **share.streamlit.io** → New app → select repo/branch/`app.py`.
3. Add your Odds API key under **App Settings → Secrets**:
   ```toml
   ODDS_API_KEY = "your_api_key_here"
   ```
4. Click **Deploy**.
    """)


# ── Sidebar navigation ────────────────────────────────────────────────────────

def main() -> None:
    # Gate: show setup page if trained model files are absent
    if not _model_files_present():
        page_setup()
        return

    cfg = load_config()

    # Load .env ODDS_API_KEY into os.environ if present (local dev only)
    if ENV_FILE.exists() and not os.environ.get("ODDS_API_KEY"):
        try:
            for line in ENV_FILE.read_text().splitlines():
                if line.startswith("ODDS_API_KEY="):
                    os.environ["ODDS_API_KEY"] = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass

    # Config key as fallback (local dev)
    if not os.environ.get("ODDS_API_KEY") and cfg.get("odds_api_key"):
        os.environ["ODDS_API_KEY"] = cfg["odds_api_key"]

    with st.sidebar:
        st.markdown(
            "<h2 style='margin-bottom:4px'>🎾 Tennis<br>Predictions</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Ensemble model · XGB + LR + MLP")
        st.divider()

        page = st.radio(
            "Navigation",
            options=["Daily Slate", "Match Deep Dive", "Model Performance", "Settings"],
            label_visibility="collapsed",
        )
        st.divider()

        # Quick stats
        preds = load_predictions()
        if not preds.empty:
            today_preds = preds[preds["date"].dt.date == date.today()] if "date" in preds.columns else preds
            st.caption(f"Today: {len(today_preds)} predictions")
            st.caption(f"Total: {len(preds):,}")

        if not _data_present():
            st.caption("ℹ️ Running in cloud mode — Match Deep Dive requires local data.")

    if page == "Daily Slate":
        page_daily_slate(cfg)
    elif page == "Match Deep Dive":
        page_deep_dive(cfg)
    elif page == "Model Performance":
        page_performance(cfg)
    elif page == "Settings":
        page_settings(cfg)


if __name__ == "__main__":
    main()
