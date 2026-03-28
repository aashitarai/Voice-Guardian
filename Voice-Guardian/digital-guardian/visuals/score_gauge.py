import plotly.graph_objects as go
import numpy as np


def render_trust_gauge(trust_score: float, verdict: str, risk: str) -> go.Figure:
    """
    Render a circular gauge showing the Trust Score (0–100%).
    Red zone: 0–44 (Likely AI)
    Orange zone: 45–69 (Uncertain)
    Green zone: 70–100 (Likely Human)
    """
    if trust_score >= 70:
        bar_color = "#22c55e"
    elif trust_score >= 45:
        bar_color = "#f97316"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=trust_score,
            number={
                "suffix": "%",
                "font": {"size": 48, "color": bar_color},
            },
            title={
                "text": f"<b>TRUST SCORE</b><br><span style='font-size:18px;color:{bar_color}'>{verdict}</span><br><span style='font-size:13px;color:#aaa'>{risk}</span>",
                "font": {"size": 20, "color": "white"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 2,
                    "tickcolor": "#555",
                    "tickfont": {"color": "#aaa"},
                },
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#1e2130",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "#3b0000"},
                    {"range": [45, 70], "color": "#3b2200"},
                    {"range": [70, 100], "color": "#003b10"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": trust_score,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#0e1117",
        font={"color": "white", "family": "Inter, sans-serif"},
        height=360,
        margin=dict(l=30, r=30, t=20, b=20),
    )

    return fig


def render_component_bars(component_scores: dict) -> go.Figure:
    """
    Horizontal bar chart showing each feature's individual Trust Score contribution.
    """
    labels = list(component_scores.keys())
    values = list(component_scores.values())

    colors = []
    for v in values:
        if v >= 70:
            colors.append("#22c55e")
        elif v >= 45:
            colors.append("#f97316")
        else:
            colors.append("#ef4444")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=12),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_vline(x=45, line_dash="dot", line_color="#f97316", opacity=0.6)
    fig.add_vline(x=70, line_dash="dot", line_color="#22c55e", opacity=0.6)

    fig.update_layout(
        title=dict(
            text="Feature Analysis Breakdown",
            font=dict(color="white", size=15),
            x=0,
        ),
        xaxis=dict(
            range=[0, 110],
            title_font=dict(color="#aaa"),
            tickfont=dict(color="#aaa"),
            gridcolor="#333",
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="white", size=13),
            gridcolor="#333",
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=300,
        margin=dict(l=10, r=60, t=50, b=40),
        showlegend=False,
    )

    return fig
