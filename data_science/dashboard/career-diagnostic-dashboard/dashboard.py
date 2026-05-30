import streamlit as st

from src.data_loader import load_data
from src.components import load_css
from src.pages.market_overview import render_market_overview
from src.pages.role_analysis import render_role_analysis
from src.pages.skill_intelligence import render_skill_intelligence


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TechJobScope — Indonesia Tech Career Intelligence",
    page_icon="""<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 24 24"><g fill="none" stroke="#000000" stroke-width="1.5"><path stroke-linecap="round" d="M11.007 21.5H9.605c-3.585 0-5.377 0-6.491-1.135S2 17.403 2 13.75s0-5.48 1.114-6.615S6.02 6 9.605 6h3.803c3.585 0 5.378 0 6.492 1.135c.857.873 1.054 2.156 1.1 4.365V13"/><path stroke-linecap="round" d="M19 18.5h-3m0 3a3 3 0 1 1 0-6m3 6a3 3 0 1 0 0-6"/><path d="m16 6l-.1-.31c-.495-1.54-.742-2.31-1.331-2.75c-.59-.44-1.372-.44-2.938-.44h-.263c-1.565 0-2.348 0-2.937.44c-.59.44-.837 1.21-1.332 2.75L7 6"/></g></svg>""",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
edges_df, kb_df, jobs_df = load_data()

ROLE_LIST = sorted(jobs_df["job_category"].dropna().unique().tolist())


# ─────────────────────────────────────────────
# SIDEBAR FILTER
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:1.1rem; font-weight:800; 
             background:linear-gradient(135deg,#60a5fa,#06b6d4);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            TechJobScope
        </div>
        <div style='font-size:0.72rem; color:#475569; margin-top:2px;'>
            Indonesia Tech Career Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown("**Global Filters**")

    selected_roles = st.multiselect(
        "Filter by Role",
        options=ROLE_LIST,
        default=ROLE_LIST,
        key="global_role",
    )

    selected_source = st.multiselect(
        "Filter by Market",
        options=["Global", "Indonesia"],
        default=["Global", "Indonesia"],
        key="global_source",
    )

    search_skill = st.text_input(
        "🔍 Search Skill",
        placeholder="e.g. Laravel, React, AWS..."
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; line-height:1.6;'>
        <b style='color:#475569;'>📊 Dataset</b><br>
        8,644 job postings<br>
        7 role categories<br>
        LinkedIn × Jobstreet<br>
        <br>
        <b style='color:#475569;'>🗓 Last Updated</b><br>
        2024 Data Snapshot
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
if not selected_roles:
    selected_roles = ROLE_LIST

if not selected_source:
    selected_source = ["Global", "Indonesia"]

jobs_filtered = jobs_df[
    (jobs_df["job_category"].isin(selected_roles)) &
    (jobs_df["market_source"].isin(selected_source))
].copy()


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Job Market Overview",
    "Role Analysis",
    "Skill Intelligence",
])


with tab1:
    render_market_overview(
        jobs_filtered=jobs_filtered,
        search_skill=search_skill
    )


with tab2:
    render_role_analysis(
        jobs_df=jobs_df,
        kb_df=kb_df,
        selected_source=selected_source,
        role_list=ROLE_LIST
    )


with tab3:
    render_skill_intelligence(
        jobs_filtered=jobs_filtered,
        edges_df=edges_df,
        selected_roles=selected_roles
    )