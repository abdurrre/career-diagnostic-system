import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

from src.config import PLOTLY_TEMPLATE, ROLE_COLORS


def render_market_overview(jobs_filtered, search_skill):
    st.markdown('<div class="page-title">Job Market Overview</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="page-subtitle">Helicopter view of 8,644 tech job postings across the global & Indonesia market.</div>', unsafe_allow_html=True)

    # KPI Row
    total_jobs = len(jobs_filtered)
    top_role = jobs_filtered["job_category"].value_counts().idxmax() if total_jobs > 0 else "—"
    top_role_cnt = jobs_filtered["job_category"].value_counts().max() if total_jobs > 0 else 0

    all_skills = []
    for s in jobs_filtered["cleaned_skills"].dropna():
        all_skills.extend([x.strip() for x in s.split(",")])
    unique_skills = len(set(all_skills))

    indo_jobs   = jobs_filtered[jobs_filtered["market_source"] == "Indonesia"]
    global_jobs = jobs_filtered[jobs_filtered["market_source"] == "Global"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_jobs:,}</div>
            <div class="kpi-label">Total Job Postings</div>
            <div class="kpi-sub">Across all selected roles</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">#{1}</div>
            <div class="kpi-label">Top Demanded Role</div>
            <div class="kpi-sub">{top_role} · {top_role_cnt:,} postings</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{unique_skills:,}</div>
            <div class="kpi-label">Unique Skills Detected</div>
            <div class="kpi-sub">Across all filtered postings</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        avg_skills = (
            jobs_filtered["cleaned_skills"]
            .dropna()
            .apply(
                lambda x: len(set(
                    s.strip().lower()
                    for s in x.split(",")
                    if s.strip()
                ))
            )
            .mean()
        )

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_skills:.1f}</div>
            <div class="kpi-label">AVG SKILLS PER JOB</div>
            <div class="kpi-sub">
                Average unique skills required per job posting
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns([1, 1.4])
    
    with col_left:
        st.markdown('<div class="section-header">Role Category Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Share of job demand across 7 tech professions</div>', unsafe_allow_html=True)

        role_counts = jobs_filtered["job_category"].value_counts().reset_index()
        role_counts.columns = ["role", "count"]
        
        fig_donut = go.Figure(go.Pie(
            labels=role_counts["role"],
            values=role_counts["count"],
            hole=0.55,
            marker=dict(colors=ROLE_COLORS, line=dict(color="#0d1117", width=2)),
            textinfo="percent",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,} jobs<br>%{percent}<extra></extra>",
            pull=[0.04 if r == top_role else 0 for r in role_counts["role"]],
        ))
        
        fig_donut.add_annotation(
            text=f"<b>{total_jobs:,}</b><br><span style='font-size:10px'>Total Jobs</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#e2e8f0"),
        )
        
        fig_donut.update_layout(**PLOTLY_TEMPLATE)
        
        fig_donut.update_layout(
            **PLOTLY_TEMPLATE,
            showlegend=True,
            legend=dict(
                orientation="v", x=1.02, y=0.5,
                font=dict(size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(t=10, b=10, l=10, r=120),
            height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        # Count role distribution
        market_role = (
            jobs_filtered
            .groupby(["market_source", "job_category"])
            .size()
            .reset_index(name="count")
        )

        # Normalize INSIDE each market
        market_role["pct"] = (
            market_role
            .groupby("market_source")["count"]
            .transform(lambda x: x / x.sum() * 100)
        )

        # Pivot
        pivot_df = (
            market_role
            .pivot(
                index="market_source",
                columns="job_category",
                values="pct"
            )
            .fillna(0)
        )

        # Role order
        role_order = (
            jobs_filtered["job_category"]
            .value_counts()
            .index
            .tolist()
        )

        pivot_df = pivot_df.reindex(columns=role_order)

        # Colors
        role_colors = {
            "AI / Machine Learning Engineer": "#8b5cf6",
            "Backend Developer": "#3b82f6",
            "Data Analyst": "#06b6d4",
            "Data Engineer": "#10b981",
            "Data Scientist": "#f59e0b",
            "Frontend Developer": "#ef4444",
            "Fullstack Developer": "#ec4899",
        }

        st.markdown(
            '<div class="section-header">Role Distribution within Each Market (%)</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-sub">Shows which job roles dominate each market after normalizing within the market itself</div>',
            unsafe_allow_html=True
        )

        # Count role distribution
        market_role = (
            jobs_filtered
            .groupby(["market_source", "job_category"])
            .size()
            .reset_index(name="count")
        )

        # Normalize inside each market
        market_role["pct"] = (
            market_role
            .groupby("market_source")["count"]
            .transform(lambda x: x / x.sum() * 100)
        )

        # Pivot table
        pivot_df = (
            market_role
            .pivot(
                index="market_source",
                columns="job_category",
                values="pct"
            )
            .fillna(0)
        )

        # Role order
        role_order = (
            jobs_filtered["job_category"]
            .value_counts()
            .index
            .tolist()
        )

        pivot_df = pivot_df.reindex(columns=role_order)

        # Color palette
        role_colors = {
            "AI / Machine Learning Engineer": "#8b5cf6",
            "Backend Developer": "#3b82f6",
            "Data Analyst": "#06b6d4",
            "Data Engineer": "#10b981",
            "Data Scientist": "#f59e0b",
            "Frontend Developer": "#ef4444",
            "Fullstack Developer": "#ec4899",
        }

        fig_market = go.Figure()

        for role in role_order:

            if role not in pivot_df.columns:
                continue

            values = pivot_df[role]

            text_values = values.apply(
                lambda x: f"{x:.1f}%" if x >= 4 else ""
            )

            fig_market.add_trace(go.Bar(
                x=pivot_df.index,
                y=values,

                name=role,

                marker=dict(
                    color=role_colors.get(role, "#64748b"),
                    line=dict(
                        color="rgba(15,23,42,0.9)",
                        width=1
                    )
                ),

                text=text_values,
                textposition="inside",

                textfont=dict(
                    size=11,
                    color="white"
                ),

                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + role +
                    ": %{y:.1f}%<extra></extra>"
                )
            ))

        fig_market.update_layout(
            **PLOTLY_TEMPLATE,

            barmode="stack",

            height=500,

            margin=dict(
                t=30,
                b=40,
                l=40,
                r=20
            ),

            xaxis=dict(
                title="Market Source",
                tickfont=dict(size=13)
            ),

            yaxis=dict(
                title="Role Composition (%)",
                range=[0, 100],
                ticksuffix="%"
            ),

            legend=dict(
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center",
                font=dict(size=11)
            ),

            uniformtext=dict(
                mode="hide",
                minsize=10
            )
        )

        st.plotly_chart(
            fig_market,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # — Search skill feature —
    if search_skill.strip():
        st.markdown(f'<div class="section-header">🔍 Search Results: "{search_skill}"</div>', unsafe_allow_html=True)
        skill_lower = search_skill.lower().strip()

        skill_hits = jobs_filtered[
            jobs_filtered["cleaned_skills"].str.contains(skill_lower, case=False, na=False)
        ]
        by_role = skill_hits.groupby("job_category").size().reset_index(name="count").sort_values("count", ascending=True)

        if len(by_role) > 0:
            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{len(skill_hits):,}</div>
                    <div class="kpi-label">Jobs Requiring "{search_skill}"</div>
                    <div class="kpi-sub">{round(len(skill_hits)/total_jobs*100,1)}% of filtered postings</div>
                </div>""", unsafe_allow_html=True)

            with col_s2:
                fig_search = go.Figure(go.Bar(
                    x=by_role["count"], y=by_role["job_category"],
                    orientation="h",
                    marker=dict(
                        color=by_role["count"],
                        colorscale=[[0,"#1e3a5f"],[1,"#3b82f6"]],
                        line_width=0,
                    ),
                    hovertemplate="<b>%{y}</b><br>%{x:,} jobs<extra></extra>",
                    text=by_role["count"],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=11),
                ))
                fig_search.update_layout(
                    **PLOTLY_TEMPLATE,
                    height=220,
                    margin=dict(t=10, b=10, l=10, r=60),
                    xaxis=dict(title="Job Count"),
                )
                st.plotly_chart(fig_search, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info(f'No jobs found containing skill: **"{search_skill}"**')

    # — Top skills overall —
    st.markdown('<div class="section-header">Top 15 Most In-Demand Skills</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Aggregated skill frequency from cleaned job postings</div>', unsafe_allow_html=True)

    from collections import Counter
    skill_counter = Counter()
    for s in jobs_filtered["cleaned_skills"].dropna():
        skill_counter.update([x.strip() for x in s.split(",") if x.strip()])

    top_skills_df = pd.DataFrame(skill_counter.most_common(15), columns=["skill", "count"])

    fig_top = go.Figure(go.Bar(
        y=top_skills_df["skill"][::-1],
        x=top_skills_df["count"][::-1],
        orientation="h",
        marker=dict(
            color=top_skills_df["count"][::-1],
            colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#06b6d4"]],
            line_width=0,
        ),
        hovertemplate="<b>%{y}</b><br>%{x:,} mentions<extra></extra>",
        text=top_skills_df["count"][::-1],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_top.update_layout(
        **PLOTLY_TEMPLATE,
        height=380,
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(title="Frequency"),
    )
    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})

    # Export
    csv_export = top_skills_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export Top Skills to CSV",
        data=csv_export,
        file_name="top_skills_market_pulse.csv",
        mime="text/csv",
    )