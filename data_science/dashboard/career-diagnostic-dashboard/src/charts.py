import plotly.graph_objects as go
import numpy as np

from src.config import PLOTLY_TEMPLATE, ROLE_COLORS, TIER_COLORS


def create_skill_heatmap(heatmap_matrix):
    fig = go.Figure(
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

    fig.update_layout(
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

    return fig