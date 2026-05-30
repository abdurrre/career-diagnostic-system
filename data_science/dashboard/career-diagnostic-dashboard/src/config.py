TIER_COLORS = {
    "Critical": "#ef4444",
    "Important": "#f59e0b",
    "Supplementary": "#22c55e",
}

ROLE_COLORS = [
    "#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b",
    "#ef4444", "#22c55e", "#ec4899",
]

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter",
        color="#94a3b8",
        size=12
    ),
    colorway=ROLE_COLORS,
)