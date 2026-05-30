import streamlit as st
import plotly.graph_objects as go

from src.config import PLOTLY_TEMPLATE, TIER_COLORS


def render_role_analysis(jobs_df, kb_df, selected_source, role_list):
    st.markdown('<div class="page-title">Role Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep-dive into the skill landscape of each tech profession. Pick a role!!</div>', unsafe_allow_html=True)

    # Role selector
    sel_role = st.selectbox(
        "Select Role to Analyse",
        options=role_list,
        index=0,
        key="skill_radar_role",
    )

    role_kb    = kb_df[kb_df["job_category"] == sel_role].copy()
    role_jobs  = jobs_df[
        (jobs_df["job_category"] == sel_role) &
        (jobs_df["market_source"].isin(selected_source))
    ]

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Top 10 skills bar chart
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown('<div class="section-header">Top 10 Critical Skills by Frequency</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Color-coded by tier: Critical → Important → Supplementary</div>', unsafe_allow_html=True)

        top10 = role_kb.sort_values("frequency", ascending=False).head(10)

        fig_hbar = go.Figure()
        for tier, color in TIER_COLORS.items():
            sub = top10[top10["tier"] == tier]
            if sub.empty:
                continue
            fig_hbar.add_trace(go.Bar(
                y=sub["skill"],
                x=sub["frequency"],
                name=tier,
                orientation="h",
                marker=dict(color=color, opacity=0.85, line_width=0),
                hovertemplate="<b>%{y}</b><br>Frequency: %{x:,}<br>Tier: " + tier + "<extra></extra>",
                text=sub["frequency"],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
            ))

        fig_hbar.update_layout(
            **PLOTLY_TEMPLATE,
            barmode="overlay",
            height=380,
            margin=dict(t=20, b=20, l=10, r=80),
            xaxis=dict(title="Frequency"),
            yaxis=dict(categoryorder="total ascending"),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_hbar, use_container_width=True, config={"displayModeBar": False})

        # Tier legend pills
        st.markdown("""
        <div style='display:flex; gap:10px; margin-top:-8px;'>
            <span class="badge-critical">🔴 Critical</span>
            <span class="badge-important">🟡 Important</span>
            <span class="badge-supplementary">🟢 Supplementary</span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Skills per Job Posting Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">How many skills does each job posting demand?</div>', unsafe_allow_html=True)

        skill_counts_per_job = role_jobs["cleaned_skills"].dropna().apply(
            lambda x: len(set(
                s.strip().lower()
                for s in x.split(",")
                if s.strip()
            ))
        )

        fig_box = go.Figure()
        fig_box.add_trace(go.Violin(
            y=skill_counts_per_job,
            box_visible=True,
            line_color="#3b82f6",
            fillcolor="rgba(59,130,246,0.12)",
            meanline_visible=True,
            meanline_color="#06b6d4",
            name=sel_role,
            hoverinfo="y",
        ))

        fig_box.update_layout(
            **PLOTLY_TEMPLATE,
            height=380,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(title="Skills per Posting"),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

        # Quick stats
        if len(skill_counts_per_job) > 0:
            st.markdown(f"""
            <div style='display:flex; gap:12px; flex-wrap:wrap; margin-top:4px;'>
                <div style='background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.25); border-radius:10px; padding:8px 14px; text-align:center;'>
                    <div style='font-size:1.2rem; font-weight:700; color:#60a5fa;'>{skill_counts_per_job.mean():.1f}</div>
                    <div style='font-size:0.7rem; color:#64748b;'>Mean</div>
                </div>
                <div style='background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.25); border-radius:10px; padding:8px 14px; text-align:center;'>
                    <div style='font-size:1.2rem; font-weight:700; color:#22d3ee;'>{skill_counts_per_job.median():.0f}</div>
                    <div style='font-size:0.7rem; color:#64748b;'>Median</div>
                </div>
                <div style='background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.25); border-radius:10px; padding:8px 14px; text-align:center;'>
                    <div style='font-size:1.2rem; font-weight:700; color:#a78bfa;'>{skill_counts_per_job.max():.0f}</div>
                    <div style='font-size:0.7rem; color:#64748b;'>Max</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    
    avg_skills_by_role = (
    jobs_df.dropna(subset=["cleaned_skills"])
        .assign(
            skill_count=lambda df: df["cleaned_skills"].apply(
                lambda x: len(set(
                    s.strip().lower()
                    for s in x.split(",")
                    if s.strip()
                ))
            )
        )
        .groupby("job_category")["skill_count"]
        .mean()
        .reset_index()
        .sort_values("skill_count", ascending=True)
    )

    fig_avg_skill = go.Figure(go.Bar(
        x=avg_skills_by_role["skill_count"],
        y=avg_skills_by_role["job_category"],
        orientation="h",
        marker=dict(
            color=avg_skills_by_role["skill_count"],
            colorscale=[
                [0, "#1e3a5f"],
                [0.5, "#2563eb"],
                [1, "#06b6d4"]
            ],
            line_width=0,
        ),
        text=avg_skills_by_role["skill_count"].round(1),
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>Average skills: %{x:.1f}<extra></extra>",
    ))

    fig_avg_skill.update_layout(
        **PLOTLY_TEMPLATE,
        height=380,
        margin=dict(t=20, b=20, l=10, r=80),
        xaxis=dict(title="Average Number of Skills"),
        yaxis=dict(title="Profession"),
    )

    st.markdown('<div class="section-header">Average Skills Required per Profession</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Average number of unique skills requested in each job posting by profession</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(fig_avg_skill, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    # Export skill table
    st.markdown('<div class="section-header">Full Skill Table</div>', unsafe_allow_html=True)

    display_kb = role_kb.sort_values("frequency", ascending=False).reset_index(drop=True)
    display_kb.index += 1

    # Color tier column
    def style_tier(val):
        colors = {"Critical": "#fee2e2", "Important": "#fef3c7", "Supplementary": "#dcfce7"}
        bg = colors.get(val, "transparent")
        txt = {"Critical": "#b91c1c", "Important": "#92400e", "Supplementary": "#166534"}
        return f"background-color:{bg}; color:{txt.get(val,'#000')}; border-radius:4px; padding:2px 8px; font-weight:600; font-size:0.78rem;"

    st.dataframe(
        display_kb[["skill", "frequency", "rank_in_role", "tier"]].style.applymap(
            style_tier, subset=["tier"]
        ),
        use_container_width=True,
        height=280,
    )

    csv_skill = role_kb.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️  Export {sel_role} Skills to CSV",
        data=csv_skill,
        file_name=f"skills_{sel_role.replace(' ','_').lower()}.csv",
        mime="text/csv",
    )