import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.config import PLOTLY_TEMPLATE
from src.preprocessing import (
    build_skill_role_matrix,
    build_normalized_market_skill_comparison,
)


def render_skill_intelligence(jobs_filtered, edges_df, selected_roles):
    st.markdown('<div class="page-title">Skill Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Skill Frequency Heatmap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Compare how often top skills appear across different job roles</div>',
        unsafe_allow_html=True
    )

    top_n_heatmap = st.slider(
        "Top skills to show in heatmap",
        min_value=10,
        max_value=30,
        value=20,
        step=5,
        key="heatmap_top_n"
    )

    heatmap_matrix = build_skill_role_matrix(jobs_filtered, top_n_heatmap)

    if not heatmap_matrix.empty:
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_matrix.values,
                x=heatmap_matrix.columns,
                y=heatmap_matrix.index,
                text=heatmap_matrix.values.astype(int),
                texttemplate="%{text}",
                textfont=dict(size=10, color="#e2e8f0"),
                colorscale=[
                    [0, "#0f172a"],
                    [0.5, "#2563eb"],
                    [1, "#06b6d4"]
                ],
                hovertemplate=(
                    "<b>Role:</b> %{y}<br>"
                    "<b>Skill:</b> %{x}<br>"
                    "<b>Frequency:</b> %{z:,}<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(
                        text="Frequency",
                        font=dict(color="#64748b", size=11)
                    ),
                    tickfont=dict(color="#64748b"),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(99,179,237,0.15)",
                    thickness=12
                )
            )
        )

        fig_heatmap.update_layout(
            **PLOTLY_TEMPLATE,
            height=520,
            margin=dict(t=20, b=120, l=120, r=30),
            xaxis=dict(
                title="Skill",
                tickangle=-45,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title="Job Role",
                tickfont=dict(size=11)
            )
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
            config={"displayModeBar": False}
        )
    else:
        st.info("No skill data available for heatmap.")
    
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # — Network graph —
    st.markdown('<div class="section-header">Skill Co-occurrence Network</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Skills connected by lines are frequently demanded together. Thicker = stronger relationship. Hover nodes for details.</div>', unsafe_allow_html=True)

    selected_network_role = st.selectbox(
        "Select role for network graph",
        options=selected_roles,
        index=0,
        key="network_role"
    )

    role_edges = edges_df[
        edges_df["role_category"] == selected_network_role
    ].copy()
    
    # Build network using plotly scatter
    top_n_nodes = st.slider("Max skills to show in network", 10, 30, 18, key="net_slider")

    # Get top N skills by total weight
    node_weights = {}
    for _, row in role_edges.iterrows():
        node_weights[row["source_skill"]] = node_weights.get(row["source_skill"], 0) + row["weight"]
        node_weights[row["target_skill"]] = node_weights.get(row["target_skill"], 0) + row["weight"]

    top_nodes = sorted(node_weights.items(), key=lambda x: -x[1])[:top_n_nodes]
    top_node_set = {n[0] for n in top_nodes}

    net_edges = role_edges[
        role_edges["source_skill"].isin(top_node_set) &
        role_edges["target_skill"].isin(top_node_set)
    ].copy()

    if not net_edges.empty:
        # Circular layout
        nodes = list(top_node_set)
        n = len(nodes)
        angles = [2 * np.pi * i / n for i in range(n)]
        node_pos = {nd: (np.cos(a), np.sin(a)) for nd, a in zip(nodes, angles)}

        # Edges
        edge_traces = []
        max_w = net_edges["weight"].max()
        for _, row in net_edges.iterrows():
            x0, y0 = node_pos[row["source_skill"]]
            x1, y1 = node_pos[row["target_skill"]]
            norm_w  = row["weight"] / max_w
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=norm_w * 4 + 0.5, color=f"rgba(59,130,246,{0.12 + norm_w * 0.45})"),
                hoverinfo="none",
                showlegend=False,
            ))

        # Nodes
        node_x = [node_pos[n][0] for n in nodes]
        node_y = [node_pos[n][1] for n in nodes]
        node_sizes = [max(18, min(45, node_weights.get(n, 0) / max(node_weights.values()) * 45)) for n in nodes]
        node_text  = [
            f"<b>{n}</b><br>Total co-occurrence: {node_weights.get(n,0):,}"
            for n in nodes
        ]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            marker=dict(
                size=node_sizes,
                color=[node_weights.get(n, 0) for n in nodes],
                colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#06b6d4"]],
                line=dict(width=1.5, color="rgba(255,255,255,0.15)"),
                showscale=True,
                colorbar=dict(
                    title=dict(text="Weight", font=dict(color="#64748b", size=10)),
                    tickfont=dict(color="#64748b", size=9),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(99,179,237,0.15)",
                    thickness=10,
                ),
            ),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )

        fig_net = go.Figure(data=edge_traces + [node_trace])
        fig_net.update_layout(
            **PLOTLY_TEMPLATE,
            height=520,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3]),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No network data available for this role filter combination.")
    
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-header">Normalized Skill Demand: Global vs Indonesia</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">Percentage of postings in each market that require the skill. This avoids bias from unequal dataset sizes.</div>',
        unsafe_allow_html=True
    )

    top_n_dumbbell = st.slider(
        "Top skills to compare",
        min_value=10,
        max_value=30,
        value=15,
        step=5,
        key="dumbbell_top_n"
    )

    market_compare_df = build_normalized_market_skill_comparison(
        jobs_filtered,
        top_n=top_n_dumbbell
    )
    
    chart_height = max(520, len(market_compare_df) * 45)

    if not market_compare_df.empty:
        market_compare_df = market_compare_df.sort_values("avg_pct", ascending=True)

        fig_dumbbell = go.Figure()

        # Connecting lines
        for _, row in market_compare_df.iterrows():
            fig_dumbbell.add_trace(go.Scatter(
                x=[row["Indonesia_pct"], row["Global_pct"]],
                y=[row["skill"], row["skill"]],
                mode="lines",
                line=dict(
                    color="rgba(148,163,184,0.45)",
                    width=2
                ),
                hoverinfo="skip",
                showlegend=False
            ))

        # Indonesia dots
        fig_dumbbell.add_trace(go.Scatter(
            x=market_compare_df["Indonesia_pct"],
            y=market_compare_df["skill"],
            mode="markers+text",
            name="Indonesia",
            marker=dict(
                size=12,
                color="#06b6d4",
                line=dict(width=1, color="rgba(255,255,255,0.25)")
            ),
            text=market_compare_df["Indonesia_pct"].round(1).astype(str) + "%",
            textposition="middle left",
            textfont=dict(size=10, color="#94a3b8"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Indonesia: %{x:.1f}% of postings<extra></extra>"
            )
        ))

        # Global dots
        fig_dumbbell.add_trace(go.Scatter(
            x=market_compare_df["Global_pct"],
            y=market_compare_df["skill"],
            mode="markers+text",
            name="Global",
            marker=dict(
                size=12,
                color="#3b82f6",
                line=dict(width=1, color="rgba(255,255,255,0.25)")
            ),
            text=market_compare_df["Global_pct"].round(1).astype(str) + "%",
            textposition="middle right",
            textfont=dict(size=10, color="#94a3b8"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Global: %{x:.1f}% of postings<extra></extra>"
            )
        ))

        fig_dumbbell.update_layout(
            **PLOTLY_TEMPLATE,
            height=chart_height,
            margin=dict(t=30, b=40, l=130, r=60),
            xaxis=dict(
                title="Share of Job Postings Requiring Skill (%)",
                ticksuffix="%",
                zeroline=True,
                zerolinecolor="rgba(148,163,184,0.25)"
            ),
            yaxis=dict(
                title="Skill",
                categoryorder="array",
                categoryarray=market_compare_df["skill"].tolist()
            ),
            legend=dict(
                orientation="h",
                y=1.08,
                x=0.5,
                xanchor="center"
            )
        )

        st.plotly_chart(
            fig_dumbbell,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    else:
        st.info("No normalized market comparison data available.")